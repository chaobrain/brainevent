import importlib.util
import sys
import types
import warnings
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainevent
from brainevent._fcn.binary import binary_fcnmv_p
from brainevent._misc import fixed_conn_num_to_csc

_COBA_EI_PATH = _ROOT / 'dev' / 'fcn' / 'COBA EI benchmark.py'
_MM_VRAM_LIMIT_PATH = _ROOT / 'dev' / 'fcn' / 'COBA_EI_binary_fcnmm_VRAM_limit_CsvOutput.py'
_PLATFORM = jax.default_backend()
_HAS_BINARY_JAX_RAW = 'jax_raw' in binary_fcnmv_p.available_backends(_PLATFORM)
_PLATFORM_DEVICE = jax.devices(_PLATFORM)[0]
LARGE_SCALE_VRAM_LIMIT_GB = 1
LARGE_SCALE_MM_BATCH_SIZE = 32
LARGE_SCALE_MM_SCALE = 100
LARGE_SCALE_MM_N = 4000
LARGE_SCALE_MM_DATA_SIZE = 4
LARGE_SCALE_MV_SCALE = 20
LARGE_SCALE_MV_CONN_NUM = 500

SUPPORTED_ROUTE_CASES = (
    ('binary', 'post', 'row_gather'),
    ('binary', 'pre', 'row_gather'),
    ('binary', 'pre', 'col_scatter'),
    ('bitpack', 'post', 'row_gather'),
    ('bitpack', 'pre', 'row_gather'),
    ('bitpack_a0', 'post', 'row_gather'),
    ('bitpack_a0', 'pre', 'row_gather'),
    ('bitpack_a1', 'post', 'row_gather'),
    ('bitpack_a1', 'pre', 'row_gather'),
)

COMPACT_ROUTE_CASES = (
    ('compact', 'post', 'row_gather'),
    ('compact', 'pre', 'row_gather'),
    ('compact', 'pre', 'col_scatter'),
    ('compact_only_vector', 'post', 'row_gather'),
    ('compact_only_vector', 'pre', 'row_gather'),
    ('compact_only_vector', 'pre', 'col_scatter'),
)

ROUTE_CASES = (*SUPPORTED_ROUTE_CASES, *COMPACT_ROUTE_CASES)
COMPACT_RESTORED_WARNING = (
    'Compact binary FCN operators are restored for this benchmark but remain '
    'experimental and may be removed again; prefer data_type="binary" or a '
    'bitpack data_type for stable benchmark runs.'
)

BITPACK_SPECIALIZED_MM_BACKENDS = ()

BITPACK_SPECIALIZED_MM_BACKEND_CASES = ()

SPECIALIZED_MM_BACKEND_CASES = (
    ('binary', 'post', 'row_gather', 'test_colmajor_fullwarp_nocap'),
    *BITPACK_SPECIALIZED_MM_BACKEND_CASES,
)


def ensure_brainpy_stub():
    if 'brainpy' in sys.modules:
        return
    if find_spec('brainpy') is not None:
        __import__('brainpy')
        return
    brainpy = types.ModuleType('brainpy')
    brainpy.state = types.SimpleNamespace()
    sys.modules['brainpy'] = brainpy


def has_real_brainpy() -> bool:
    return find_spec('brainpy') is not None


def load_module(module_name: str, path: Path):
    ensure_brainpy_stub()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.FixedPostNumConn = fixed_post_num_conn_cls()
    module.BinaryArray = binary_array_cls()
    module.BitPackedBinary = bitpacked_binary_cls()
    module.CompactBinary = compact_binary_cls()
    return module


def coba_ei_module():
    return load_module('coba_ei_benchmark_mod', _COBA_EI_PATH)


def mm_vram_limit_module():
    return load_module('coba_ei_mm_vram_limit_mod', _MM_VRAM_LIMIT_PATH)


@contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextmanager
def force_backend(backend: str):
    old_backend = brainevent.config.get_backend(_PLATFORM)
    brainevent.config.set_backend(_PLATFORM, backend)
    try:
        yield
    finally:
        brainevent.config.set_backend(_PLATFORM, old_backend)


def assert_quantity_allclose(actual, expected, *, rtol=1e-5, atol=1e-5):
    actual_value, actual_unit = u.split_mantissa_unit(actual)
    expected_value, expected_unit = u.split_mantissa_unit(expected)
    assert actual_unit == expected_unit
    assert jnp.allclose(
        jnp.asarray(actual_value),
        jnp.asarray(expected_value),
        rtol=rtol,
        atol=atol,
    )


def mantissa_array(x):
    return jnp.asarray(u.get_mantissa(x), dtype=brainstate.environ.dftype())


def fake_syn_decay(tau) -> float:
    tau_val = float(jnp.asarray(u.get_mantissa(tau)).reshape(()))
    return float(jnp.exp(-1.0 / max(tau_val, 1e-6)))


class FakeCOBA:
    def __init__(self, E):
        self.E = E
        self.cond = None

    def bind_cond(self, cond):
        self.cond = mantissa_array(cond)

    def current(self):
        if self.cond is None:
            return jnp.asarray(0.0, dtype=brainstate.environ.dftype())
        return self.cond


class FakeExpon:
    def __init__(self, size, tau):
        self.size = size
        self.tau = tau
        self.decay = fake_syn_decay(tau)
        self.state = jnp.zeros(size, dtype=brainstate.environ.dftype())

    def __call__(self, delta):
        self.state = self.decay * self.state + mantissa_array(delta)
        return self.state


class FakeLIFRef:
    def __init__(self, num, **kwargs):
        self.num = num
        self.inputs = {}
        self._spike = jnp.zeros(num, dtype=jnp.bool_)

    def add_current_input(self, name, obj):
        self.inputs[name] = obj

    def get_spike(self):
        return self._spike

    def __call__(self, inp):
        total = mantissa_array(inp)
        for obj in self.inputs.values():
            total = total + jnp.asarray(obj.current(), dtype=brainstate.environ.dftype())
        self._spike = total > jnp.mean(total)
        return self._spike


def install_fake_brainpy_state(module, monkeypatch):
    fake_state = types.SimpleNamespace(
        LIFRef=FakeLIFRef,
        Expon=FakeExpon,
        COBA=FakeCOBA,
    )
    monkeypatch.setattr(module.brainpy, 'state', fake_state)


def binary_array_cls():
    from brainevent._event.binary import BinaryArray
    return BinaryArray


def bitpacked_binary_cls():
    from brainevent._event.bitpack_binary import BitPackedBinary
    return BitPackedBinary


def compact_binary_cls():
    from brainevent._event.compact_binary import CompactBinary
    return CompactBinary


def fixed_post_num_conn_cls():
    from brainevent._fcn.main import FixedPostNumConn
    return FixedPostNumConn


def preferred_real_backend(data_type: str) -> str | None:
    if data_type == 'binary':
        backends = brainevent.binary_fcnmv_p.available_backends(_PLATFORM)
        if 'cuda_raw' in backends:
            return 'cuda_raw'
        if 'jax_raw' in backends:
            return 'jax_raw'
        return None
    if data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
        backends = brainevent.bitpack_binary_fcnmv_p.available_backends(_PLATFORM)
        return 'cuda_raw' if 'cuda_raw' in backends else None
    if data_type in ('compact', 'compact_only_vector'):
        backends = brainevent.compact_binary_fcnmv_p.available_backends(_PLATFORM)
        if 'cuda_raw' in backends:
            return 'cuda_raw'
        if 'numba' in backends:
            return 'numba'
        return None
    return None


def preferred_real_mm_backend(data_type: str) -> str | None:
    if data_type == 'binary':
        backends = brainevent.binary_fcnmm_p.available_backends(_PLATFORM)
        if 'cuda_raw' in backends:
            return 'cuda_raw'
        if 'jax_raw' in backends:
            return 'jax_raw'
        return None
    if data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
        backends = brainevent.bitpack_binary_fcnmm_p.available_backends(_PLATFORM)
        return 'cuda_raw' if 'cuda_raw' in backends else None
    if data_type == 'compact':
        backends = brainevent.compact_binary_fcnmm_p.available_backends(_PLATFORM)
        if 'cuda_raw' in backends:
            return 'cuda_raw'
        if 'numba' in backends:
            return 'numba'
        return None
    if data_type == 'compact_only_vector':
        return None
    return None


def spikes_from_seed(size: int, seed: int, *, p_active: float = 0.05):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random(size) < p_active, dtype=jnp.bool_)


def matrix_from_seed(
    source_size: int,
    batch_size: int,
    seed: int,
    *,
    efferent_target: str,
    p_active: float = 0.35,
):
    rng = np.random.default_rng(seed)
    if efferent_target == 'post':
        return jnp.asarray(rng.random((batch_size, source_size)) < p_active, dtype=jnp.bool_)
    return jnp.asarray(rng.random((source_size, batch_size)) < p_active, dtype=jnp.bool_)


def prepare_operand_like_coba_ei(spikes, *, data_type: str):
    # Match the real COBA EI path by materializing operand data on the
    # active JAX platform instead of pinning it to CPU.
    with jax.default_device(_PLATFORM_DEVICE):
        spikes = u.math.asarray(spikes, dtype=jnp.bool_)
        if data_type == 'binary':
            return binary_array_cls()(spikes)
        if data_type == 'compact':
            warnings.warn(COMPACT_RESTORED_WARNING, UserWarning, stacklevel=2)
            return compact_binary_cls().from_array(spikes)
        if data_type == 'compact_only_vector':
            warnings.warn(COMPACT_RESTORED_WARNING, UserWarning, stacklevel=2)
            if spikes.ndim != 1:
                raise ValueError('compact_only_vector is only supported for MV/vector inputs.')
            return compact_binary_cls().compacy_only_vector(spikes)
        if data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
            return bitpacked_binary_cls()(spikes)
    raise ValueError(f'Unsupported binary-family data_type: {data_type}')


def resolve_bitpack_mm_pack_axis_like_coba_ei(mod, data_type: str) -> int:
    resolver = getattr(mod, '_resolve_bitpack_mm_pack_axis', None)
    if resolver is not None:
        return resolver(data_type)
    return 1 if data_type == 'bitpack_a1' else 0


def build_conn_like_coba_ei(
    source_size: int,
    target_size: int,
    conn_num: int,
    *,
    data_type: str,
    efferent_target: str,
    homo: bool,
    conn_weight_base,
    mv_layout: str,
    backend: str | None = None,
):
    mod = coba_ei_module()
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
        and (
            (data_type in ('binary', 'compact') and mv_layout == 'col_scatter')
            or data_type == 'compact_only_vector'
        )
    )
    bitpack_mm_pack_axis = resolve_bitpack_mm_pack_axis_like_coba_ei(mod, data_type)

    # Keep connectivity buffers on the same platform as the operand so the
    # compact GPU preprocessing path stays on-device end-to-end.
    with jax.default_device(_PLATFORM_DEVICE):
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
        conn_cls = fixed_post_num_conn_cls()

        try:
            return conn_cls(
                (weight, indices),
                shape=shape,
                backend=backend,
                maintain_dual_layout=maintain_dual_layout,
                bitpack_mm_pack_axis=bitpack_mm_pack_axis,
            )
        except TypeError as exc:
            if (
                'maintain_dual_layout' not in str(exc)
                and 'bitpack_mm_pack_axis' not in str(exc)
                and 'backend' not in str(exc)
            ):
                raise
            conn = conn_cls((weight, indices), shape=shape)
            conn.backend = backend
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


def apply_conn_like_coba_ei(spikes, conn, *, data_type: str, efferent_target: str):
    operand = prepare_operand_like_coba_ei(spikes, data_type=data_type)
    if efferent_target == 'post':
        return operand @ conn
    return conn @ operand


def binary_jax_route_reference(
    source_size: int,
    target_size: int,
    conn_num: int,
    spikes,
    *,
    seed: int,
    efferent_target: str,
    homo: bool,
    mv_layout: str,
    conn_weight_base,
):
    with numpy_seed(seed):
        conn = build_conn_like_coba_ei(
            source_size,
            target_size,
            conn_num,
            data_type='binary',
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=conn_weight_base,
            mv_layout=mv_layout,
            backend='jax_raw',
        )
    with force_backend('jax_raw'):
        return apply_conn_like_coba_ei(
            spikes,
            conn,
            data_type='binary',
            efferent_target=efferent_target,
        )


def assert_final_run_matches_binary_jax(actual, expected, *, rtol=1e-5, atol=1e-5):
    actual_num, actual_rate = actual
    expected_num, expected_rate = expected
    assert int(actual_num) == int(expected_num)
    assert_quantity_allclose(actual_rate, expected_rate, rtol=rtol, atol=atol)


def run_simulation_once(
    mod,
    *,
    seed: int,
    scale: float,
    data_type: str,
    efferent_target: str,
    duration,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    backend: str,
):
    with force_backend(backend):
        brainstate.random.seed(seed)
        with numpy_seed(seed):
            run = mod.make_simulation_run(
                scale=scale,
                data_type=data_type,
                efferent_target=efferent_target,
                duration=duration,
                conn_num=conn_num,
                homo=homo,
                mv_layout=mv_layout,
            )
        return jax.block_until_ready(run())


def run_batch_simulation_once(
    mod,
    *,
    seed: int,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    duration,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    backend: str,
):
    with force_backend(backend):
        brainstate.random.seed(seed)
        with numpy_seed(seed):
            run = mod.make_simulation_batch_run(
                scale=scale,
                batch_size=batch_size,
                data_type=data_type,
                efferent_target=efferent_target,
                duration=duration,
                conn_num=conn_num,
                homo=homo,
                mv_layout=mv_layout,
            )
        return jax.block_until_ready(run())


def build_shared_large_scale_points():
    mod = mm_vram_limit_module()
    return mod._generate_mm_boundary_pairs(
        limit_gb=2,
        batch_size=32,
        sample_points_per_batch=2,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        homo=True,
        data_size=4,
    )


def build_large_scale_point_for_vram(*, limit_gb: int, batch_size: int):
    mod = mm_vram_limit_module()
    pairs = mod._generate_mm_boundary_pairs(
        limit_gb=limit_gb,
        batch_size=batch_size,
        sample_points_per_batch=1,
        _N=4000,
        scale_max=2000,
        conn_max=4000,
        homo=True,
        data_size=4,
    )
    if not pairs:
        raise ValueError(f'No boundary pair available for limit_gb={limit_gb}, batch_size={batch_size}.')
    return pairs[-1]


def build_shared_mm_large_scale_point():
    limit_bytes = LARGE_SCALE_VRAM_LIMIT_GB * (1024 ** 3)
    size = LARGE_SCALE_MM_SCALE * LARGE_SCALE_MM_N
    batch_elems = 2 * LARGE_SCALE_MM_BATCH_SIZE * size
    budget_elems = limit_bytes // LARGE_SCALE_MM_DATA_SIZE
    conn = int((budget_elems - batch_elems) // size)
    if conn < 1:
        raise ValueError(
            'No connection budget available for '
            f'scale={LARGE_SCALE_MM_SCALE}, batch_size={LARGE_SCALE_MM_BATCH_SIZE}.'
        )
    return LARGE_SCALE_MM_SCALE, conn


def build_shared_mv_large_scale_point():
    return LARGE_SCALE_MV_SCALE, LARGE_SCALE_MV_CONN_NUM


def assert_spike_history_matches(actual, expected, *, atol: float):
    actual_np = np.asarray(jax.device_get(actual), dtype=np.bool_)
    expected_np = np.asarray(jax.device_get(expected), dtype=np.bool_)
    assert actual_np.shape == expected_np.shape
    assert np.array_equal(actual_np, expected_np)


def assert_step_trace_matches(actual, expected, *, atol: float):
    assert actual.keys() == expected.keys()
    for name in actual:
        if name == 'spike_bits':
            actual_np = np.asarray(jax.device_get(actual[name]), dtype=np.uint8)
            expected_np = np.asarray(jax.device_get(expected[name]), dtype=np.uint8)
            assert actual_np.shape == expected_np.shape, name
            assert np.array_equal(actual_np, expected_np), name
            continue
        if name == 'spike_shape':
            actual_np = np.asarray(jax.device_get(actual[name]), dtype=np.int32)
            expected_np = np.asarray(jax.device_get(expected[name]), dtype=np.int32)
            assert np.array_equal(actual_np, expected_np), name
            continue
        actual_np = np.asarray(jax.device_get(actual[name]), dtype=np.float32)
        expected_np = np.asarray(jax.device_get(expected[name]), dtype=np.float32)
        assert actual_np.shape == expected_np.shape, name
        assert np.allclose(actual_np, expected_np, rtol=0.0, atol=atol), name


def _spike_trace_bits(spikes):
    spikes = jnp.asarray(spikes, dtype=jnp.bool_)
    return jnp.packbits(jnp.reshape(spikes, (-1,)).astype(jnp.uint8))


def _spike_trace_shape(spikes):
    return jnp.asarray(spikes.shape, dtype=jnp.int32)


def _trace_summary(x, *, sample_size: int = 32):
    x = jnp.asarray(x, dtype=brainstate.environ.dftype())
    flat = jnp.reshape(x, (-1,))
    n = flat.shape[0]
    if n <= sample_size:
        sample_ids = jnp.arange(n, dtype=jnp.int32)
    else:
        sample_ids = jnp.linspace(0, n - 1, sample_size, dtype=jnp.int32)
    return jnp.concatenate(
        [
            jnp.asarray(
                [
                    jnp.mean(flat),
                    jnp.mean(jnp.abs(flat)),
                    jnp.max(flat),
                    jnp.min(flat),
                    jnp.mean(flat != 0),
                ],
                dtype=brainstate.environ.dftype(),
            ),
            flat[sample_ids],
        ],
        axis=0,
    )


def _instantiate_shared_conn(
    data,
    indices,
    *,
    shape,
    data_type: str,
    efferent_target: str,
    mv_layout: str,
    backend: str,
):
    maintain_dual_layout = (
        efferent_target == 'pre'
        and (
            (data_type in ('binary', 'compact') and mv_layout == 'col_scatter')
            or data_type == 'compact_only_vector'
        )
    )
    bitpack_mm_pack_axis = resolve_bitpack_mm_pack_axis_like_coba_ei(coba_ei_module(), data_type)
    conn_cls = fixed_post_num_conn_cls()
    try:
        return conn_cls(
            (data, indices),
            shape=shape,
            backend=backend,
            maintain_dual_layout=maintain_dual_layout,
            bitpack_mm_pack_axis=bitpack_mm_pack_axis,
        )
    except TypeError as exc:
        if (
            'maintain_dual_layout' not in str(exc)
            and 'bitpack_mm_pack_axis' not in str(exc)
            and 'backend' not in str(exc)
        ):
            raise
        conn = conn_cls((data, indices), shape=shape)
        conn.backend = backend
        conn.bitpack_mm_pack_axis = bitpack_mm_pack_axis
        if maintain_dual_layout:
            col_weights, col_indices, col_indptr = fixed_conn_num_to_csc(data, indices, shape=shape)
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


def build_shared_conn_pair(
    source_size: int,
    target_size: int,
    conn_num: int,
    *,
    data_type: str,
    efferent_target: str,
    homo: bool,
    conn_weight_base,
    mv_layout: str,
    actual_backend: str | None,
    reference_backend: str = 'jax_raw',
    conn_seed: int = 0,
):
    with numpy_seed(conn_seed):
        base_conn = build_conn_like_coba_ei(
            source_size,
            target_size,
            conn_num,
            data_type='binary',
            efferent_target=efferent_target,
            homo=homo,
            conn_weight_base=conn_weight_base,
            mv_layout=mv_layout,
            backend=reference_backend,
        )
    data = base_conn.data
    indices = base_conn.indices
    shape = base_conn.shape
    actual_conn = _instantiate_shared_conn(
        data,
        indices,
        shape=shape,
        data_type=data_type,
        efferent_target=efferent_target,
        mv_layout=mv_layout,
        backend=actual_backend,
    )
    reference_conn = _instantiate_shared_conn(
        data,
        indices,
        shape=shape,
        data_type='binary',
        efferent_target=efferent_target,
        mv_layout=mv_layout,
        backend=reference_backend,
    )
    return actual_conn, reference_conn


def build_shared_batch_mm_conn_pair(
    source_size: int,
    target_size: int,
    conn_num: int,
    *,
    data_type: str,
    efferent_target: str,
    homo: bool,
    conn_weight_base,
    mv_layout: str,
    reference_backend: str = 'jax_raw',
    conn_seed: int = 0,
):
    """Mirror COBA EI batch benchmark backend selection for MM-style routes.

    The output scripts select the actual backend globally via
    ``brainevent.config.set_backend(...)`` and construct connections without an
    explicit backend override. Keep that same shape here so batch E2E tests and
    benchmark runs lower through the same dispatch path.
    """
    return build_shared_conn_pair(
        source_size,
        target_size,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=conn_weight_base,
        mv_layout=mv_layout,
        actual_backend=None,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )


def _run_single_net_spike_history(
    mod,
    *,
    conn_pair,
    scale: float,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    steps: int,
):
    net = mod.EINet(
        scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
    )
    exc_conn, inh_conn = conn_pair
    net.exc_conn = exc_conn
    net.inh_conn = inh_conn
    net.init_all_states()
    spikes = []
    with brainstate.environ.context(dt=0.1 * u.ms):
        times = u.math.arange(0. * u.ms, steps * brainstate.environ.get_dt(), brainstate.environ.get_dt())
        for t in times:
            spikes.append(net.update(t, 24 * u.mA, exc_conn=exc_conn, inh_conn=inh_conn))
    return jnp.stack(spikes, axis=0)


def _run_real_single_net_spike_history(
    mod,
    *,
    scale: float,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    backend: str,
    steps: int,
    conn_seed: int,
    state_seed: int,
):
    with force_backend(backend):
        brainstate.random.seed(state_seed)
        with numpy_seed(conn_seed):
            net = mod.EINet(
                scale,
                data_type=data_type,
                efferent_target=efferent_target,
                conn_num=conn_num,
                homo=homo,
                mv_layout=mv_layout,
                backend=backend,
            )
        exc_conn = net.exc_conn
        inh_conn = net.inh_conn
        net.exc_conn = None
        net.inh_conn = None

        @brainstate.transform.jit
        def run(exc_conn, inh_conn):
            net.init_all_states()

            def fn(t):
                return net.update(t, 24 * u.mA, exc_conn=exc_conn, inh_conn=inh_conn)

            with brainstate.environ.context(dt=0.1 * u.ms):
                times = u.math.arange(0. * u.ms, steps * brainstate.environ.get_dt(), brainstate.environ.get_dt())
                return brainstate.transform.for_loop(fn, times)

        return run(exc_conn, inh_conn)


def _run_single_net_step_trace(
    mod,
    *,
    conn_pair,
    scale: float,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    steps: int,
    summarize: bool,
):
    net = mod.EINet(
        scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
    )
    exc_conn, inh_conn = conn_pair
    net.exc_conn = None
    net.inh_conn = None
    net.N._spike = jnp.arange(net.num, dtype=jnp.int32) % 17 == 0

    @brainstate.transform.jit
    def run(exc_conn, inh_conn):
        net.init_all_states()

        def fn(t):
            return net.update_with_intermediates(
                t,
                24 * u.mA,
                exc_conn=exc_conn,
                inh_conn=inh_conn,
            )

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, steps * brainstate.environ.get_dt(), brainstate.environ.get_dt())
            trace = brainstate.transform.for_loop(fn, times)

        if not summarize:
            dtype = brainstate.environ.dftype()
            return {
                'spikes': jnp.asarray(trace[0], dtype=jnp.bool_),
                'delta_g_exc': jnp.asarray(trace[1], dtype=dtype),
                'delta_g_inh': jnp.asarray(trace[2], dtype=dtype),
                'g_exc': jnp.asarray(trace[3], dtype=dtype),
                'g_inh': jnp.asarray(trace[4], dtype=dtype),
                'input_current': jnp.asarray(trace[5], dtype=dtype),
            }

        return {
            'spike_bits': _spike_trace_bits(trace[0]),
            'spike_shape': _spike_trace_shape(trace[0]),
            'delta_g_exc_summary': _trace_summary(trace[1]),
            'delta_g_inh_summary': _trace_summary(trace[2]),
            'g_exc_summary': _trace_summary(trace[3]),
            'g_inh_summary': _trace_summary(trace[4]),
            'input_current_summary': _trace_summary(trace[5]),
        }

    return run(exc_conn, inh_conn)


def run_full_e2e_spike_history_once(
    mod,
    *,
    scale: float,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    actual_backend: str,
    reference_backend: str = 'jax_raw',
    steps: int = 10,
    conn_seed: int = 0,
    state_seed: int = 0,
):
    actual = _run_real_single_net_spike_history(
        mod,
        scale=scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
        backend=actual_backend,
        steps=steps,
        conn_seed=conn_seed,
        state_seed=state_seed,
    )
    expected = _run_real_single_net_spike_history(
        mod,
        scale=scale,
        data_type='binary',
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
        backend=reference_backend,
        steps=steps,
        conn_seed=conn_seed,
        state_seed=state_seed,
    )
    return actual, expected


def run_e2e_spike_history_once(
    mod,
    *,
    scale: float,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    actual_backend: str,
    reference_backend: str = 'jax_raw',
    steps: int = 10,
    conn_seed: int = 0,
    state_seed: int = 0,
):
    n_exc = int(3200 * scale)
    n_inh = int(800 * scale)
    num = n_exc + n_inh
    actual_exc_conn, reference_exc_conn = build_shared_conn_pair(
        n_exc,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=0.6 * u.mS,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )
    actual_inh_conn, reference_inh_conn = build_shared_conn_pair(
        n_inh,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=6.7 * u.mS,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )

    with force_backend(actual_backend):
        brainstate.random.seed(state_seed)
        actual = _run_single_net_spike_history(
            mod,
            conn_pair=(actual_exc_conn, actual_inh_conn),
            scale=scale,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
        )
    with force_backend(reference_backend):
        brainstate.random.seed(state_seed)
        expected = _run_single_net_spike_history(
            mod,
            conn_pair=(reference_exc_conn, reference_inh_conn),
            scale=scale,
            data_type='binary',
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
        )
    return actual, expected


def run_e2e_step_trace_once(
    mod,
    *,
    scale: float,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    actual_backend: str,
    reference_backend: str = 'jax_raw',
    steps: int = 10,
    conn_seed: int = 0,
    state_seed: int = 0,
    summarize: bool = True,
):
    n_exc = int(3200 * scale)
    n_inh = int(800 * scale)
    num = n_exc + n_inh
    actual_exc_conn, reference_exc_conn = build_shared_conn_pair(
        n_exc,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=0.6 * u.mS,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )
    actual_inh_conn, reference_inh_conn = build_shared_conn_pair(
        n_inh,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=6.7 * u.mS,
        mv_layout=mv_layout,
        actual_backend=actual_backend,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )

    with force_backend(actual_backend):
        brainstate.random.seed(state_seed)
        actual = _run_single_net_step_trace(
            mod,
            conn_pair=(actual_exc_conn, actual_inh_conn),
            scale=scale,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
            summarize=summarize,
        )
    with force_backend(reference_backend):
        brainstate.random.seed(state_seed)
        expected = _run_single_net_step_trace(
            mod,
            conn_pair=(reference_exc_conn, reference_inh_conn),
            scale=scale,
            data_type='binary',
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
            summarize=summarize,
        )
    return actual, expected


def _run_batch_net_spike_history(
    mod,
    *,
    conn_pair,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    steps: int,
):
    net = mod.EINet(
        scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
    )
    exc_conn, inh_conn = conn_pair
    net.exc_conn = exc_conn
    net.inh_conn = inh_conn
    mapper = brainstate.nn.Map(net, init_map_size=batch_size)

    @brainstate.transform.jit
    def run():
        mapper.init_all_states()

        def fn(t):
            ts = jnp.ones(batch_size) * t
            return mapper.map('update', in_axes=(0, None))(ts, 24. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, steps * brainstate.environ.get_dt(), brainstate.environ.get_dt())
            return brainstate.transform.for_loop(fn, times)

    return run()


def _run_real_batch_net_spike_history(
    mod,
    *,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    backend: str,
    steps: int,
    conn_seed: int,
    state_seed: int,
):
    with force_backend(backend):
        brainstate.random.seed(state_seed)
        with numpy_seed(conn_seed):
            net = mod.EINet(
                scale,
                data_type=data_type,
                efferent_target=efferent_target,
                conn_num=conn_num,
                homo=homo,
                mv_layout=mv_layout,
            )
        mapper = brainstate.nn.Map(net, init_map_size=batch_size)

        @brainstate.transform.jit
        def run():
            mapper.init_all_states()

            def fn(t):
                ts = jnp.ones(batch_size) * t
                return mapper.map('update', in_axes=(0, None))(ts, 24. * u.mA)

            with brainstate.environ.context(dt=0.1 * u.ms):
                times = u.math.arange(0. * u.ms, steps * brainstate.environ.get_dt(), brainstate.environ.get_dt())
                return brainstate.transform.for_loop(fn, times)

        return run()


def _run_batch_net_step_trace(
    mod,
    *,
    conn_pair,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    steps: int,
):
    net = mod.EINet(
        scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
    )
    exc_conn, inh_conn = conn_pair
    net.exc_conn = None
    net.inh_conn = None
    net.N._spike = jnp.arange(net.num, dtype=jnp.int32) % 17 == 0
    mapper = brainstate.nn.Map(net, init_map_size=batch_size)

    @brainstate.transform.jit
    def run(exc_conn, inh_conn):
        mapper.init_all_states()

        def fn(t):
            ts = jnp.ones(batch_size) * t
            return mapper.map('update_with_intermediates_mapped', in_axes=(0, None, None, None))(
                ts,
                24. * u.mA,
                exc_conn,
                inh_conn,
            )

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, steps * brainstate.environ.get_dt(), brainstate.environ.get_dt())
            trace = brainstate.transform.for_loop(fn, times)

        return {
            'spike_bits': _spike_trace_bits(trace[0]),
            'spike_shape': _spike_trace_shape(trace[0]),
            'delta_g_exc_summary': _trace_summary(trace[1]),
            'delta_g_inh_summary': _trace_summary(trace[2]),
            'g_exc_summary': _trace_summary(trace[3]),
            'g_inh_summary': _trace_summary(trace[4]),
            'input_current_summary': _trace_summary(trace[5]),
        }

    return run(exc_conn, inh_conn)


def run_full_batch_e2e_spike_history_once(
    mod,
    *,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    actual_backend: str,
    reference_backend: str = 'jax_raw',
    steps: int = 10,
    conn_seed: int = 0,
    state_seed: int = 0,
):
    actual = _run_real_batch_net_spike_history(
        mod,
        scale=scale,
        batch_size=batch_size,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
        backend=actual_backend,
        steps=steps,
        conn_seed=conn_seed,
        state_seed=state_seed,
    )
    expected = _run_real_batch_net_spike_history(
        mod,
        scale=scale,
        batch_size=batch_size,
        data_type='binary',
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
        backend=reference_backend,
        steps=steps,
        conn_seed=conn_seed,
        state_seed=state_seed,
    )
    return actual, expected


def run_batch_e2e_spike_history_once(
    mod,
    *,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    actual_backend: str,
    reference_backend: str = 'jax_raw',
    steps: int = 10,
    conn_seed: int = 0,
    state_seed: int = 0,
):
    n_exc = int(3200 * scale)
    n_inh = int(800 * scale)
    num = n_exc + n_inh
    actual_exc_conn, reference_exc_conn = build_shared_batch_mm_conn_pair(
        n_exc,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=0.6 * u.mS,
        mv_layout=mv_layout,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )
    actual_inh_conn, reference_inh_conn = build_shared_batch_mm_conn_pair(
        n_inh,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=6.7 * u.mS,
        mv_layout=mv_layout,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )

    with force_backend(actual_backend):
        brainstate.random.seed(state_seed)
        actual = _run_batch_net_spike_history(
            mod,
            conn_pair=(actual_exc_conn, actual_inh_conn),
            scale=scale,
            batch_size=batch_size,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
        )
    with force_backend(reference_backend):
        brainstate.random.seed(state_seed)
        expected = _run_batch_net_spike_history(
            mod,
            conn_pair=(reference_exc_conn, reference_inh_conn),
            scale=scale,
            batch_size=batch_size,
            data_type='binary',
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
        )
    return actual, expected


def run_batch_e2e_step_trace_once(
    mod,
    *,
    scale: float,
    batch_size: int,
    data_type: str,
    efferent_target: str,
    conn_num: int,
    homo: bool,
    mv_layout: str,
    actual_backend: str,
    reference_backend: str = 'jax_raw',
    steps: int = 10,
    conn_seed: int = 0,
    state_seed: int = 0,
):
    n_exc = int(3200 * scale)
    n_inh = int(800 * scale)
    num = n_exc + n_inh
    actual_exc_conn, reference_exc_conn = build_shared_batch_mm_conn_pair(
        n_exc,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=0.6 * u.mS,
        mv_layout=mv_layout,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )
    actual_inh_conn, reference_inh_conn = build_shared_batch_mm_conn_pair(
        n_inh,
        num,
        conn_num,
        data_type=data_type,
        efferent_target=efferent_target,
        homo=homo,
        conn_weight_base=6.7 * u.mS,
        mv_layout=mv_layout,
        reference_backend=reference_backend,
        conn_seed=conn_seed,
    )

    with force_backend(actual_backend):
        brainstate.random.seed(state_seed)
        actual = _run_batch_net_step_trace(
            mod,
            conn_pair=(actual_exc_conn, actual_inh_conn),
            scale=scale,
            batch_size=batch_size,
            data_type=data_type,
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
        )
    with force_backend(reference_backend):
        brainstate.random.seed(state_seed)
        expected = _run_batch_net_step_trace(
            mod,
            conn_pair=(reference_exc_conn, reference_inh_conn),
            scale=scale,
            batch_size=batch_size,
            data_type='binary',
            efferent_target=efferent_target,
            conn_num=conn_num,
            homo=homo,
            mv_layout=mv_layout,
            steps=steps,
        )
    return actual, expected
