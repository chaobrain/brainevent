import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._fcn.main as fcn_main_mod
from brainevent._fcn.binary import binary_fcnmv_p
from brainevent._fcn.float import fcnmv, fcnmm
from brainevent._misc import fixed_conn_num_to_csc

_COBA_EI_PATH = _ROOT / 'dev' / 'fcn' / 'COBA EI benchmark.py'
_COBA_2005_PATH = _ROOT / 'dev' / 'fcn' / 'COBA_2005_benchmark.py'
_PLATFORM = jax.default_backend()
_HAS_BINARY_JAX_RAW = 'jax_raw' in binary_fcnmv_p.available_backends(_PLATFORM)
_CPU_DEVICE = jax.devices('cpu')[0]

_ROUTE_CASES = (
    pytest.param('binary', 'post', 'row_gather', id='binary-post'),
    pytest.param('binary', 'pre', 'row_gather', id='binary-pre-row-gather'),
    pytest.param('binary', 'pre', 'col_scatter', id='binary-pre-col-scatter'),
    pytest.param('compact', 'post', 'row_gather', id='compact-post'),
    pytest.param('compact', 'pre', 'row_gather', id='compact-pre-row-gather'),
    pytest.param('compact', 'pre', 'col_scatter', id='compact-pre-col-scatter'),
    pytest.param('bitpack', 'post', 'row_gather', id='bitpack-post'),
    pytest.param('bitpack', 'pre', 'row_gather', id='bitpack-pre'),
    pytest.param('bitpack_a0', 'post', 'row_gather', id='bitpack-a0-post'),
    pytest.param('bitpack_a0', 'pre', 'row_gather', id='bitpack-a0-pre'),
    pytest.param('bitpack_a1', 'post', 'row_gather', id='bitpack-a1-post'),
    pytest.param('bitpack_a1', 'pre', 'row_gather', id='bitpack-a1-pre'),
    pytest.param('float', 'post', 'row_gather', id='float-post'),
    pytest.param('float', 'pre', 'row_gather', id='float-pre'),
)

_BINARY_ROUTE_CASES = (
    pytest.param('binary', 'post', 'row_gather', id='binary-post'),
    pytest.param('binary', 'pre', 'row_gather', id='binary-pre-row-gather'),
    pytest.param('binary', 'pre', 'col_scatter', id='binary-pre-col-scatter'),
)


def _ensure_brainpy_stub():
    if 'brainpy' in sys.modules:
        return
    brainpy = types.ModuleType('brainpy')
    brainpy.state = types.SimpleNamespace()
    sys.modules['brainpy'] = brainpy


def _load_module(module_name: str, path: Path):
    _ensure_brainpy_stub()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # Keep the dev benchmark helpers aligned with the current workspace
    # implementations even when the module is loaded via importlib.
    module.FixedPostNumConn = _fixed_post_num_conn_cls()
    module.BinaryArray = _binary_array_cls()
    module.BitPackedBinary = _bitpacked_binary_cls()
    module.CompactBinary = _compact_binary_cls()
    return module


def _coba_ei_module():
    return _load_module('coba_ei_benchmark_mod', _COBA_EI_PATH)


def _coba_2005_module():
    return _load_module('coba_2005_benchmark_mod', _COBA_2005_PATH)


@contextmanager
def _numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextmanager
def _force_backend(backend: str):
    old_backend = brainevent.config.get_backend(_PLATFORM)
    brainevent.config.set_backend(_PLATFORM, backend)
    try:
        yield
    finally:
        brainevent.config.set_backend(_PLATFORM, old_backend)


def _assert_quantity_allclose(actual, expected, *, rtol=1e-5, atol=1e-5):
    actual_value, actual_unit = u.split_mantissa_unit(actual)
    expected_value, expected_unit = u.split_mantissa_unit(expected)
    assert actual_unit == expected_unit
    assert jnp.allclose(
        jnp.asarray(actual_value),
        jnp.asarray(expected_value),
        rtol=rtol,
        atol=atol,
    )


def _spikes_from_seed(size: int, seed: int, *, p_active: float = 0.05):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random(size) < p_active, dtype=jnp.bool_)


def _binary_array_cls():
    from brainevent._event.binary import BinaryArray
    return BinaryArray


def _bitpacked_binary_cls():
    from brainevent._event.bitpack_binary import BitPackedBinary
    return BitPackedBinary


def _compact_binary_cls():
    from brainevent._event.compact_binary import CompactBinary
    return CompactBinary


def _fixed_post_num_conn_cls():
    from brainevent._fcn.main import FixedPostNumConn
    return FixedPostNumConn


def _compact_only_ctor():
    compact_binary_cls = _compact_binary_cls()
    ctor = getattr(compact_binary_cls, 'compacy_only_vector', None)
    if ctor is None:
        ctor = getattr(compact_binary_cls, 'compact_only_vector', None)
    return ctor


def _prepare_operand_like_coba_ei(spikes, *, data_type: str, efferent_target: str):
    spikes = u.math.asarray(spikes, dtype=jnp.bool_)
    if data_type == 'binary':
        return _binary_array_cls()(spikes)
    if data_type == 'compact':
        if efferent_target == 'post':
            compact_only_ctor = _compact_only_ctor()
            if compact_only_ctor is not None:
                return compact_only_ctor(spikes)
            return _compact_binary_cls().from_array(spikes)
        return _compact_binary_cls().from_array(spikes)
    if data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
        return _bitpacked_binary_cls()(spikes)
    if data_type == 'float':
        return u.math.asarray(spikes, dtype=brainstate.environ.dftype())
    raise ValueError(f'Unsupported data_type: {data_type}')


def _resolve_bitpack_mm_pack_axis_like_coba_ei(mod, data_type: str) -> int:
    resolver = getattr(mod, '_resolve_bitpack_mm_pack_axis', None)
    if resolver is not None:
        return resolver(data_type)
    return 1 if data_type == 'bitpack_a1' else 0


def _build_conn_like_coba_ei(
    source_size: int,
    target_size: int,
    conn_num: int,
    *,
    data_type: str,
    efferent_target: str,
    homo: bool,
    conn_weight_base,
    mv_layout: str,
):
    mod = _coba_ei_module()
    total_conn_num = conn_num
    resolved_conn_num = mod._resolve_conn_num(
        conn_num,
        source_size,
        target_size,
        efferent_target=efferent_target,
    )
    if efferent_target == 'post':
        shape = (source_size, target_size)
        n_rows, n_cols = source_size, target_size
    else:
        shape = (target_size, source_size)
        n_rows, n_cols = target_size, source_size
    maintain_dual_layout = (
        efferent_target == 'pre'
        and data_type in ('binary', 'compact')
        and mv_layout == 'col_scatter'
    )
    bitpack_mm_pack_axis = _resolve_bitpack_mm_pack_axis_like_coba_ei(mod, data_type)
    indices_np = np.random.randint(0, n_cols, size=(n_rows, resolved_conn_num)).astype(np.int32, copy=False)
    conn_weight = conn_weight_base * mod.conn_num_base / total_conn_num
    if homo:
        weight = u.math.asarray(conn_weight, dtype=brainstate.environ.dftype())
    else:
        weight = u.math.asarray(
            u.math.full((n_rows, resolved_conn_num), conn_weight),
            dtype=brainstate.environ.dftype(),
        )
    indices = u.math.asarray(indices_np, dtype=np.int32)
    fixed_post_num_conn_cls = _fixed_post_num_conn_cls()
    try:
        return fixed_post_num_conn_cls(
            (weight, indices),
            shape=shape,
            maintain_dual_layout=maintain_dual_layout,
            bitpack_mm_pack_axis=bitpack_mm_pack_axis,
        )
    except TypeError as exc:
        if (
            'maintain_dual_layout' not in str(exc)
            and 'bitpack_mm_pack_axis' not in str(exc)
        ):
            raise
        conn = fixed_post_num_conn_cls((weight, indices), shape=shape)
        conn.bitpack_mm_pack_axis = bitpack_mm_pack_axis
        if maintain_dual_layout:
            col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(weight, indices, shape=shape)
            conn.maintain_dual_layout = True
            if hasattr(conn, 'set_buffer'):
                conn.set_buffer('col_weights', col_weights)
                conn.set_buffer('col_indices', col_indices)
                conn.set_buffer('col_indptr', col_indptr)
            else:
                setattr(conn, 'col_weights', col_weights)
                setattr(conn, 'col_indices', col_indices)
                setattr(conn, 'col_indptr', col_indptr)
        return conn


def _apply_conn_like_coba_ei(spikes, conn, *, data_type: str, efferent_target: str):
    operand = _prepare_operand_like_coba_ei(
        spikes,
        data_type=data_type,
        efferent_target=efferent_target,
    )
    if efferent_target == 'post':
        return operand @ conn
    return conn @ operand


def _jax_reference(conn, spikes, *, efferent_target: str):
    spikes = jnp.asarray(spikes, dtype=brainstate.environ.dftype())
    return fcnmv(
        conn.data,
        conn.indices,
        spikes,
        shape=conn.shape,
        transpose=(efferent_target == 'post'),
    )


def _dense_reference(conn, spikes, *, efferent_target: str):
    spikes = jnp.asarray(spikes)
    dense = conn.todense()
    active = jnp.asarray(spikes > 0, dtype=u.get_mantissa(dense).dtype)
    if efferent_target == 'post':
        return active @ dense
    return dense @ active


def _bitpack_reference_mv(weights, indices, packed, spikes, *, shape, transpose=False):
    del packed
    active = jnp.asarray(jnp.asarray(spikes) > 0, dtype=brainstate.environ.dftype())
    return fcnmv(weights, indices, active, shape=shape, transpose=transpose)


def _bitpack_reference_mm(weights, indices, packed, matrix, *, shape, transpose=False, pack_axis=1):
    del packed, pack_axis
    active = jnp.asarray(jnp.asarray(matrix) > 0, dtype=brainstate.environ.dftype())
    return fcnmm(weights, indices, active, shape=shape, transpose=transpose)


def _bitpacked_transpose_host(self, *axes):
    if not axes:
        perm = tuple(reversed(range(self.ndim)))
    elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        perm = tuple(axes[0])
    else:
        perm = tuple(axes)
    value = jnp.asarray(np.transpose(np.asarray(self.value), perm), dtype=self.value.dtype)
    return _bitpacked_binary_cls()(value)


def _build_explicit_conn_pair(
    *,
    seed: int,
    scale: int,
    conn_num: int,
    data_type: str,
    efferent_target: str,
    mv_layout: str,
    homo: bool,
):
    n_exc = int(3200 * scale)
    n_inh = int(800 * scale)
    total = n_exc + n_inh
    with _numpy_seed(seed):
        exc_conn = _build_conn_like_coba_ei(
            n_exc,
            total,
            conn_num,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
        )
        inh_conn = _build_conn_like_coba_ei(
            n_inh,
            total,
            conn_num,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=6.7 * u.mS,
            mv_layout=mv_layout,
        )
    return exc_conn, inh_conn, n_exc, n_inh, total


def _build_legacy_conn_pair(
    *,
    seed: int,
    scale: int,
    conn_num: int,
    data_type: str,
    efferent_target: str,
    mv_layout: str,
    homo: bool,
):
    mod = _coba_2005_module()
    n_exc = int(3200 * scale)
    n_inh = int(800 * scale)
    total = n_exc + n_inh
    with _numpy_seed(seed):
        exc_conn = mod.FixedNumConn(
            n_exc,
            total,
            conn_num=conn_num,
            efferent_target=efferent_target,
            data_type=data_type,
            homo=homo,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
        )
        inh_conn = mod.FixedNumConn(
            n_inh,
            total,
            conn_num=conn_num,
            efferent_target=efferent_target,
            data_type=data_type,
            homo=homo,
            conn_weight_base=6.7 * u.mS,
            mv_layout=mv_layout,
    )
    return exc_conn, inh_conn, n_exc, n_inh, total


def _mantissa_array(x):
    return jnp.asarray(u.get_mantissa(x), dtype=brainstate.environ.dftype())


def _fake_syn_decay(tau) -> float:
    tau_val = float(jnp.asarray(u.get_mantissa(tau)).reshape(()))
    return float(jnp.exp(-1.0 / max(tau_val, 1e-6)))


class _FakeCOBA:
    def __init__(self, E):
        self.E = E
        self.cond = None

    def bind_cond(self, cond):
        self.cond = _mantissa_array(cond)

    def current(self):
        if self.cond is None:
            return jnp.asarray(0.0, dtype=brainstate.environ.dftype())
        return self.cond


class _FakeExpon:
    def __init__(self, size, tau):
        self.size = size
        self.tau = tau
        self.decay = _fake_syn_decay(tau)
        self.state = jnp.zeros(size, dtype=brainstate.environ.dftype())

    def __call__(self, delta):
        self.state = self.decay * self.state + _mantissa_array(delta)
        return self.state


class _FakeLIFRef:
    def __init__(self, num, **kwargs):
        self.num = num
        self.inputs = {}
        self._spike = jnp.zeros(num, dtype=jnp.bool_)

    def add_current_input(self, name, obj):
        self.inputs[name] = obj

    def get_spike(self):
        return self._spike

    def __call__(self, inp):
        total = _mantissa_array(inp)
        for obj in self.inputs.values():
            total = total + jnp.asarray(obj.current(), dtype=brainstate.environ.dftype())
        self._spike = total > jnp.mean(total)
        return self._spike


def _install_fake_brainpy_state(module, monkeypatch):
    fake_state = types.SimpleNamespace(
        LIFRef=_FakeLIFRef,
        Expon=_FakeExpon,
        COBA=_FakeCOBA,
    )
    monkeypatch.setattr(module.brainpy, 'state', fake_state)


def _simulate_einet_reference(net, initial_spike, *, steps: int, inp):
    prev_spike = jnp.asarray(initial_spike, dtype=jnp.bool_)
    exc_state = jnp.zeros(net.num, dtype=brainstate.environ.dftype())
    inh_state = jnp.zeros(net.num, dtype=brainstate.environ.dftype())
    rate = jnp.zeros(net.num, dtype=brainstate.environ.dftype())
    step_spikes = []
    exc_decay = _fake_syn_decay(5.0 * u.ms)
    inh_decay = _fake_syn_decay(10.0 * u.ms)
    inp_value = _mantissa_array(inp)

    for _ in range(steps):
        exc_spk = prev_spike[:net.n_exc]
        inh_spk = prev_spike[net.n_exc:]
        delta_g_exc = _jax_reference(net.exc_conn, exc_spk, efferent_target=net.efferent_target)
        delta_g_inh = _jax_reference(net.inh_conn, inh_spk, efferent_target=net.efferent_target)
        exc_state = exc_decay * exc_state + _mantissa_array(delta_g_exc)
        inh_state = inh_decay * inh_state + _mantissa_array(delta_g_inh)
        total = inp_value + exc_state + inh_state
        prev_spike = total > jnp.mean(total)
        step_spikes.append(prev_spike)
        rate = rate + prev_spike.astype(rate.dtype)

    return step_spikes, rate, exc_state, inh_state


def test_apply_conn_uses_expected_matmul_side(monkeypatch):
    mod = _coba_ei_module()
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
    ('data_type', 'efferent_target', 'expected_kind', 'expect_compact_only'),
    [
        ('binary', 'post', 'binary', False),
        ('binary', 'pre', 'binary', False),
        ('compact', 'post', 'compact', True),
        ('compact', 'pre', 'compact', False),
        ('bitpack', 'post', 'bitpack', False),
        ('bitpack_a0', 'pre', 'bitpack', False),
        ('bitpack_a1', 'post', 'bitpack', False),
        ('float', 'pre', 'float', False),
    ],
)
def test_prepare_operand_selects_expected_representation(
    data_type,
    efferent_target,
    expected_kind,
    expect_compact_only,
):
    spikes = jnp.asarray([1, 0, 1, 0, 1], dtype=jnp.bool_)
    operand = _prepare_operand_like_coba_ei(spikes, data_type=data_type, efferent_target=efferent_target)

    if expected_kind == 'float':
        assert isinstance(operand, jax.Array)
        assert operand.dtype == brainstate.environ.dftype()
        return

    if expected_kind == 'binary':
        assert isinstance(operand, _binary_array_cls())
    elif expected_kind == 'compact':
        assert isinstance(operand, _compact_binary_cls())
    else:
        assert isinstance(operand, _bitpacked_binary_cls())

    if expected_kind == 'compact':
        if expect_compact_only and _compact_only_ctor() is not None:
            assert operand.packed.shape == (0,)
        else:
            assert operand.packed.size > 0


@pytest.mark.parametrize(
    ('data_type', 'efferent_target', 'mv_layout', 'expect_dual_layout'),
    [
        ('binary', 'pre', 'col_scatter', True),
        ('binary', 'pre', 'row_gather', False),
        ('binary', 'pre', 'auto', False),
        ('binary', 'post', 'col_scatter', False),
        ('compact', 'pre', 'col_scatter', True),
        ('compact', 'pre', 'row_gather', False),
        ('compact', 'post', 'col_scatter', False),
        ('bitpack', 'pre', 'col_scatter', False),
        ('float', 'pre', 'col_scatter', False),
    ],
)
def test_make_post_conn_enables_dual_layout_only_when_needed(
    data_type,
    efferent_target,
    mv_layout,
    expect_dual_layout,
):
    with _numpy_seed(0):
        conn = _build_conn_like_coba_ei(
            7,
            11,
            3,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
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
        ('float', 0),
        ('bitpack', 0),
        ('bitpack_a0', 0),
        ('bitpack_a1', 1),
        ('compact', 0),
    ],
)
def test_make_post_conn_assigns_expected_bitpack_mm_pack_axis(
    data_type,
    expected_pack_axis,
):
    with _numpy_seed(0):
        conn = _build_conn_like_coba_ei(
            7,
            11,
            3,
            data_type=data_type,
            efferent_target='post',
            homo=True,
            conn_weight_base=0.6 * u.mS,
            mv_layout='row_gather',
        )

    assert conn.bitpack_mm_pack_axis == expected_pack_axis


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), _ROUTE_CASES)
@pytest.mark.parametrize('homo', [True, False])
def test_each_route_matches_sparse_jax_reference_on_small_shapes(
    data_type,
    efferent_target,
    mv_layout,
    homo,
):
    source_size = 17
    target_size = 23
    with _numpy_seed(3):
        conn = _build_conn_like_coba_ei(
            source_size,
            target_size,
            5,
            data_type=data_type,
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
        )

    spikes = _spikes_from_seed(source_size, 5, p_active=0.35)
    actual = _apply_conn_like_coba_ei(spikes, conn, data_type=data_type, efferent_target=efferent_target)
    expected = _jax_reference(conn, spikes, efferent_target=efferent_target)

    jax.block_until_ready(u.get_mantissa(actual))
    jax.block_until_ready(u.get_mantissa(expected))
    _assert_quantity_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), _ROUTE_CASES)
def test_scale10_conn800_all_routes_match_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
):
    seed = 42
    scale = 10
    conn_num = 800
    exc_conn, inh_conn, n_exc, n_inh, _ = _build_explicit_conn_pair(
        seed=seed,
        scale=scale,
        conn_num=conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        mv_layout=mv_layout,
        homo=True,
    )
    exc_spikes = _spikes_from_seed(n_exc, seed, p_active=0.02)
    inh_spikes = _spikes_from_seed(n_inh, seed + 1, p_active=0.03)

    exc_actual = _apply_conn_like_coba_ei(exc_spikes, exc_conn, data_type=data_type, efferent_target=efferent_target)
    inh_actual = _apply_conn_like_coba_ei(inh_spikes, inh_conn, data_type=data_type, efferent_target=efferent_target)
    exc_expected = _jax_reference(exc_conn, exc_spikes, efferent_target=efferent_target)
    inh_expected = _jax_reference(inh_conn, inh_spikes, efferent_target=efferent_target)

    jax.block_until_ready(u.get_mantissa(exc_actual))
    jax.block_until_ready(u.get_mantissa(inh_actual))
    jax.block_until_ready(u.get_mantissa(exc_expected))
    jax.block_until_ready(u.get_mantissa(inh_expected))

    _assert_quantity_allclose(exc_actual, exc_expected, rtol=1e-4, atol=1e-4)
    _assert_quantity_allclose(inh_actual, inh_expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ('data_type', 'efferent_target'),
    [
        pytest.param('bitpack', 'post', id='bitpack-post-mm'),
        pytest.param('bitpack', 'pre', id='bitpack-pre-mm'),
        pytest.param('bitpack_a0', 'post', id='bitpack-a0-post-mm'),
        pytest.param('bitpack_a0', 'pre', id='bitpack-a0-pre-mm'),
        pytest.param('bitpack_a1', 'post', id='bitpack-a1-post-mm'),
        pytest.param('bitpack_a1', 'pre', id='bitpack-a1-pre-mm'),
    ],
)
def test_bitpack_matrix_routes_match_dense_reference(
    data_type,
    efferent_target,
    monkeypatch,
):
    monkeypatch.setattr(fcn_main_mod, 'bitpack_binary_fcnmm', _bitpack_reference_mm)
    monkeypatch.setattr(brainevent.BitPackedBinary, 'transpose', _bitpacked_transpose_host)
    source_size = 17
    target_size = 23
    batch_size = 5
    with jax.default_device(_CPU_DEVICE):
        with _numpy_seed(3):
            conn = _build_conn_like_coba_ei(
                source_size,
                target_size,
                5,
                data_type=data_type,
                efferent_target=efferent_target,
                homo=True,
                conn_weight_base=0.6 * u.mS,
                mv_layout='row_gather',
            )

        rng = np.random.default_rng(11)
        if efferent_target == 'post':
            spikes = jnp.asarray(rng.random((batch_size, source_size)) < 0.35, dtype=jnp.bool_)
        else:
            spikes = jnp.asarray(rng.random((source_size, batch_size)) < 0.35, dtype=jnp.bool_)

        actual = _apply_conn_like_coba_ei(
            spikes,
            conn,
            data_type=data_type,
            efferent_target=efferent_target,
        )
        expected = _dense_reference(conn, spikes, efferent_target=efferent_target)

    jax.block_until_ready(u.get_mantissa(actual))
    jax.block_until_ready(u.get_mantissa(expected))
    _assert_quantity_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), _ROUTE_CASES)
def test_einet_update_full_flow_matches_jax_reference(
    data_type,
    efferent_target,
    mv_layout,
    monkeypatch,
):
    mod = _coba_ei_module()
    _install_fake_brainpy_state(mod, monkeypatch)

    seed = 7
    scale = 0.01
    conn_num = 8
    steps = 4
    inp = 24.0 * u.mA

    brainstate.random.seed(seed)
    with _numpy_seed(seed):
        net = mod.EINet(
            scale=scale,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=True,
            mv_layout=mv_layout,
        )

    net.init_state()
    initial_spike = jnp.asarray((jnp.arange(net.num) % 3) == 0, dtype=jnp.bool_)
    net.N._spike = initial_spike

    expected_spikes, expected_rate, expected_exc_state, expected_inh_state = _simulate_einet_reference(
        net,
        initial_spike,
        steps=steps,
        inp=inp,
    )

    actual_spikes = []
    for step in range(steps):
        actual_spikes.append(net.update(step * 0.1 * u.ms, inp))

    jax.block_until_ready((
        actual_spikes,
        expected_spikes,
        net.rate.value,
        expected_rate,
        net.exc_syn.state,
        expected_exc_state,
        net.inh_syn.state,
        expected_inh_state,
    ))

    for actual, expected in zip(actual_spikes, expected_spikes):
        assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(net.rate.value, expected_rate)
    assert jnp.allclose(net.exc_syn.state, expected_exc_state, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(net.inh_syn.state, expected_inh_state, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('data_type', ['bitpack', 'bitpack_a1'])
def test_make_simulation_batch_run_executes_for_bitpack_routes(monkeypatch, data_type):
    mod = _coba_ei_module()
    _install_fake_brainpy_state(mod, monkeypatch)
    monkeypatch.setattr(fcn_main_mod, 'bitpack_binary_fcnmv', _bitpack_reference_mv)
    monkeypatch.setattr(fcn_main_mod, 'bitpack_binary_fcnmm', _bitpack_reference_mm)

    with jax.default_device(_CPU_DEVICE):
        brainstate.random.seed(5)
        with _numpy_seed(5):
            run = mod.make_simulation_batch_run(
                scale=0.01,
                batch_size=2,
                data_type=data_type,
                efferent_target='post',
                duration=0.3 * u.ms,
                conn_num=8,
                homo=True,
                mv_layout='row_gather',
            )

        num, rate = run()
    jax.block_until_ready((num, rate))

    assert int(num) > 0
    assert jnp.all(jnp.isfinite(jnp.asarray(rate)))


def test_training_entrypoints_remain_unimplemented():
    mod = _coba_ei_module()

    with pytest.raises(NotImplementedError, match='not implemented'):
        mod.make_training_run(scale=0.01)()
    with pytest.raises(NotImplementedError, match='not implemented'):
        mod.make_training_batch_run(scale=0.01, batch_size=2)()


@pytest.mark.skipif(
    not _HAS_BINARY_JAX_RAW,
    reason=f'binary jax_raw backend is unavailable on platform={_PLATFORM}',
)
@pytest.mark.parametrize(('data_type', 'efferent_target', 'mv_layout'), _BINARY_ROUTE_CASES)
def test_scale10_conn800_binary_routes_match_legacy_jax_raw_baseline(
    data_type,
    efferent_target,
    mv_layout,
):
    seed = 42
    scale = 10
    conn_num = 800
    exc_conn, inh_conn, n_exc, n_inh, _ = _build_explicit_conn_pair(
        seed=seed,
        scale=scale,
        conn_num=conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        mv_layout=mv_layout,
        homo=True,
    )
    legacy_exc_conn, legacy_inh_conn, _, _, _ = _build_legacy_conn_pair(
        seed=seed,
        scale=scale,
        conn_num=conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        mv_layout=mv_layout,
        homo=True,
    )
    exc_spikes = _spikes_from_seed(n_exc, seed, p_active=0.02)
    inh_spikes = _spikes_from_seed(n_inh, seed + 1, p_active=0.03)

    with _force_backend('jax_raw'):
        exc_actual = _apply_conn_like_coba_ei(
            exc_spikes,
            exc_conn,
            data_type=data_type,
            efferent_target=efferent_target,
        )
        inh_actual = _apply_conn_like_coba_ei(
            inh_spikes,
            inh_conn,
            data_type=data_type,
            efferent_target=efferent_target,
        )
        exc_expected = legacy_exc_conn.update(exc_spikes)
        inh_expected = legacy_inh_conn.update(inh_spikes)

    jax.block_until_ready(u.get_mantissa(exc_actual))
    jax.block_until_ready(u.get_mantissa(inh_actual))
    jax.block_until_ready(u.get_mantissa(exc_expected))
    jax.block_until_ready(u.get_mantissa(inh_expected))

    _assert_quantity_allclose(exc_actual, exc_expected, rtol=1e-4, atol=1e-4)
    _assert_quantity_allclose(inh_actual, inh_expected, rtol=1e-4, atol=1e-4)
