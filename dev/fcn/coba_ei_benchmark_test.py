# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib.util
import itertools
import sys
import types
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import brainunit as u
import jax.numpy as jnp
import pytest

import brainevent._fcn.main as fcn_main_mod
from brainevent._event.bitpack_binary import bitpack
from brainevent._event.compact import _compact_1d_jax
from brainevent._event.compact_binary import CompactBinary


def _install_brainpy_stub(monkeypatch):
    if importlib.util.find_spec('brainpy') is not None:
        return

    brainpy = types.ModuleType('brainpy')
    brainpy.state = types.SimpleNamespace(
        LIFRef=object,
        AlignPostProj=object,
        Expon=types.SimpleNamespace(desc=lambda *args, **kwargs: None),
        COBA=types.SimpleNamespace(desc=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, 'brainpy', brainpy)


def _install_benchmark_tools_stub(monkeypatch):
    fake_bt = types.ModuleType('BenchmarkTools')
    fake_bt.init_kwargs = []
    fake_bt.filter_calls = []
    fake_bt.filtered_states = None
    fake_bt.generate_calls = []

    class _FakeTestingParamsGeneratorMV:
        def __init__(self, *args, **kwargs):
            fake_bt.init_kwargs.append(dict(kwargs))

        def make_simulation_params_probs(self, *args, **kwargs):
            return []

        def generate_params(self, *args, **kwargs):
            fake_bt.generate_calls.append((args, dict(kwargs)))
            if fake_bt.filtered_states is None:
                return [(1, None, 20)]
            return list(fake_bt.filtered_states)

        def filter_existing_states(self, states):
            state_list = list(states)
            fake_bt.filter_calls.append(state_list)
            if fake_bt.filtered_states is None:
                return state_list
            return list(fake_bt.filtered_states)

    class _FakeCSVRecord:
        def __init__(self, *args, **kwargs):
            pass

        def print_header(self, *args, **kwargs):
            pass

        def print_table_header(self, *args, **kwargs):
            pass

        def add_tag(self, *args, **kwargs):
            pass

        def print_row(self, *args, **kwargs):
            pass

        def single_COBA_data_add(self, *args, **kwargs):
            pass

        def flush_and_clear(self, *args, **kwargs):
            return '/tmp/fake.csv'

    fake_bt.TestingParamsGenerator_mv = _FakeTestingParamsGeneratorMV
    fake_bt.CSV_record = _FakeCSVRecord
    monkeypatch.setitem(sys.modules, 'BenchmarkTools', fake_bt)
    return fake_bt


def _load_coba_ei_csv_module(monkeypatch):
    _install_brainpy_stub(monkeypatch)
    wrapper_path = Path(__file__).resolve().with_name('COBA_EI_binary_fcnmv_CsvOuput.py')
    spec = importlib.util.spec_from_file_location('coba_ei_csv_test_module', wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dense_from_fixed_conn(weights, indices, shape):
    weight_value, weight_unit = u.split_mantissa_unit(weights)
    rows = jnp.repeat(jnp.arange(shape[0], dtype=indices.dtype), indices.shape[1])
    cols = indices.reshape(-1)
    weight_value = jnp.asarray(weight_value)
    if weight_value.size == 1:
        values = jnp.full((indices.size,), weight_value.reshape(-1)[0], dtype=weight_value.dtype)
    else:
        values = weight_value.reshape(-1)
    dense = jnp.zeros(shape, dtype=weight_value.dtype).at[rows, cols].add(values)
    return u.maybe_decimal(dense * weight_unit)


def _mv_reference(weights, indices, spikes, shape, transpose):
    dense = _dense_from_fixed_conn(weights, indices, shape)
    dense_value, dense_unit = u.split_mantissa_unit(dense)
    active = jnp.asarray(spikes > 0, dtype=dense_value.dtype)
    if transpose:
        result = active @ dense_value
    else:
        result = dense_value @ active
    return u.maybe_decimal(result * dense_unit)


def _compact_from_array_jax_impl(cls, x, bit_width=32):
    x = jnp.asarray(x)
    assert x.ndim == 1
    packed = bitpack(x, axis=0)
    active_ids, n_active = _compact_1d_jax(x, jax_impl=True)
    return cls(packed, active_ids, n_active, x, n_orig=x.shape[0], batch_size=None, bit_width=bit_width)


def _compact_only_vector_jax_impl(cls, x):
    x = jnp.asarray(x)
    assert x.ndim == 1
    active_ids, n_active = _compact_1d_jax(x, jax_impl=True)
    packed = jnp.zeros((0,), dtype=jnp.uint32)
    return cls(packed, active_ids, n_active, x, n_orig=x.shape[0], batch_size=None, bit_width=32)


_DATA_TYPES = ('binary', 'float', 'bitpack', 'bitpack_a0', 'bitpack_a1', 'compact')
_MODES = ('pre', 'post')
_MV_LAYOUTS = ('col_scatter', 'row_gather', 'auto')
_ROUTE_CASES = tuple(itertools.product(_DATA_TYPES, _MODES, _MV_LAYOUTS))


def _expected_terminal_name(data_type: str) -> str:
    if data_type == 'binary':
        return 'main.binary_fcnmv'
    if data_type == 'float':
        return 'main.fcnmv'
    if data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
        return 'main.bitpack_binary_fcnmv'
    if data_type == 'compact':
        return 'main.compact_binary_fcnmv'
    raise AssertionError(f'Unexpected data_type: {data_type}')


def _expected_col_scatter(data_type: str, mode: str, mv_layout: str) -> bool:
    return data_type in ('binary', 'compact') and mode == 'pre' and mv_layout == 'col_scatter'


def _expected_compact_ctor_name(data_type: str, mode: str):
    if data_type != 'compact':
        return None
    if mode == 'post':
        return 'benchmark.CompactBinary.compacy_only_vector'
    return 'benchmark.CompactBinary.from_array'


def test_coba_ei_resolve_conn_num_integer_pre_splits_by_source_pool_ratio(monkeypatch):
    csv_mod = _load_coba_ei_csv_module(monkeypatch)
    benchmark_mod = csv_mod._BENCHMARK_MODULE

    assert benchmark_mod._resolve_conn_num(80, 3200, 4000, efferent_target='pre') == 64
    assert benchmark_mod._resolve_conn_num(80, 800, 4000, efferent_target='pre') == 16
    assert benchmark_mod._resolve_conn_num(80, 3200, 4000, efferent_target='post') == 80


def test_coba_ei_make_post_conn_integer_pre_uses_split_fixed_indegrees(monkeypatch):
    csv_mod = _load_coba_ei_csv_module(monkeypatch)
    benchmark_mod = csv_mod._BENCHMARK_MODULE

    exc_conn = benchmark_mod._make_post_conn(
        3200,
        4000,
        80,
        efferent_target='pre',
        data_type='binary',
        homo=True,
        conn_weight_base=0.6 * u.mS,
        mv_layout='row_gather',
    )
    inh_conn = benchmark_mod._make_post_conn(
        800,
        4000,
        80,
        efferent_target='pre',
        data_type='binary',
        homo=True,
        conn_weight_base=6.7 * u.mS,
        mv_layout='row_gather',
    )
    post_conn = benchmark_mod._make_post_conn(
        3200,
        4000,
        80,
        efferent_target='post',
        data_type='binary',
        homo=True,
        conn_weight_base=0.6 * u.mS,
        mv_layout='row_gather',
    )

    assert exc_conn.indices.shape == (4000, 64)
    assert inh_conn.indices.shape == (4000, 16)
    assert post_conn.indices.shape == (3200, 80)
    assert exc_conn.shape == (4000, 3200)
    assert inh_conn.shape == (4000, 800)
    assert post_conn.shape == (3200, 4000)
    exc_weight_value, exc_unit = u.split_mantissa_unit(exc_conn.data)
    inh_weight_value, inh_unit = u.split_mantissa_unit(inh_conn.data)
    post_weight_value, post_unit = u.split_mantissa_unit(post_conn.data)
    assert exc_weight_value.shape == ()
    assert inh_weight_value.shape == ()
    assert post_weight_value.shape == ()
    assert float(exc_weight_value) == pytest.approx(0.6)
    assert float(inh_weight_value) == pytest.approx(6.7)
    assert float(post_weight_value) == pytest.approx(0.6)
    assert exc_unit == u.mS
    assert inh_unit == u.mS
    assert post_unit == u.mS


@pytest.mark.parametrize(('data_type', 'mode', 'mv_layout'), _ROUTE_CASES)
def test_coba_ei_csv_wrapper_route_matrix(
    monkeypatch,
    data_type,
    mode,
    mv_layout,
):
    _install_benchmark_tools_stub(monkeypatch)
    csv_mod = _load_coba_ei_csv_module(monkeypatch)
    csv_mod.scales = [1]

    announce_calls = []
    backend_calls = []
    forwarded_calls = []
    terminal_calls = []
    call_chain = []
    capture_chain = {'enabled': True}

    expected_transpose = (mode == 'post')
    expect_col_scatter = _expected_col_scatter(data_type, mode, mv_layout)
    expected_terminal = _expected_terminal_name(data_type)

    def _record(name: str):
        if capture_chain['enabled']:
            call_chain.append(name)

    def _announce():
        announce_calls.append('called')
        return 'cpu'

    def _set_backend(platform, backend):
        backend_calls.append((platform, backend))

    def _make_terminal_spy(name: str):
        def _spy(weights, indices, *args, shape, transpose=False, **kwargs):
            _record(name)
            call = dict(kwargs)
            call['shape'] = shape
            call['transpose'] = transpose
            terminal_calls.append((name, call))
            spikes = args[-1]
            capture_chain['enabled'] = False
            return _mv_reference(weights, indices, spikes, shape, transpose)

        return _spy

    def _matmul_spy(self, other):
        _record('main.FixedPostNumConn.__matmul__')
        return original_matmul(self, other)

    def _rmatmul_spy(self, other):
        _record('main.FixedPostNumConn.__rmatmul__')
        return original_rmatmul(self, other)

    def _make_post_conn_spy(*args, **kwargs):
        _record('benchmark._make_post_conn')
        return original_make_post_conn(*args, **kwargs)

    def _prepare_operand_spy(spikes, *, data_type, efferent_target):
        _record('benchmark._prepare_operand')
        return original_prepare_operand(spikes, data_type=data_type, efferent_target=efferent_target)

    def _apply_conn_spy(spikes, conn, *, data_type, efferent_target):
        _record('benchmark._apply_conn')
        return original_apply_conn(spikes, conn, data_type=data_type, efferent_target=efferent_target)

    def _compact_ctor_spy(cls, x, bit_width=32):
        _record('benchmark.CompactBinary.from_array')
        return _compact_from_array_jax_impl(cls, x, bit_width=bit_width)

    def _compact_only_ctor_spy(cls, x):
        _record('benchmark.CompactBinary.compacy_only_vector')
        return _compact_only_vector_jax_impl(cls, x)

    original_matmul = fcn_main_mod.FixedPostNumConn.__matmul__
    original_rmatmul = fcn_main_mod.FixedPostNumConn.__rmatmul__
    original_make_post_conn = csv_mod._BENCHMARK_MODULE._make_post_conn
    original_prepare_operand = csv_mod._BENCHMARK_MODULE._prepare_operand
    original_apply_conn = csv_mod._BENCHMARK_MODULE._apply_conn

    monkeypatch.setattr(csv_mod, '_announce_runtime_platform', _announce)
    monkeypatch.setattr(csv_mod.brainevent.config, 'set_backend', _set_backend)
    monkeypatch.setattr(fcn_main_mod.FixedPostNumConn, '__matmul__', _matmul_spy)
    monkeypatch.setattr(fcn_main_mod.FixedPostNumConn, '__rmatmul__', _rmatmul_spy)
    monkeypatch.setattr(csv_mod._BENCHMARK_MODULE, '_make_post_conn', _make_post_conn_spy)
    monkeypatch.setattr(csv_mod._BENCHMARK_MODULE, '_prepare_operand', _prepare_operand_spy)
    monkeypatch.setattr(csv_mod._BENCHMARK_MODULE, '_apply_conn', _apply_conn_spy)
    monkeypatch.setattr(
        fcn_main_mod,
        'binary_fcnmv',
        _make_terminal_spy('main.binary_fcnmv'),
    )
    monkeypatch.setattr(
        fcn_main_mod,
        'compact_binary_fcnmv',
        _make_terminal_spy('main.compact_binary_fcnmv'),
    )
    monkeypatch.setattr(
        fcn_main_mod,
        'bitpack_binary_fcnmv',
        _make_terminal_spy('main.bitpack_binary_fcnmv'),
    )
    monkeypatch.setattr(
        fcn_main_mod,
        'fcnmv',
        _make_terminal_spy('main.fcnmv'),
    )
    monkeypatch.setattr(csv_mod._BENCHMARK_MODULE.CompactBinary, 'from_array', classmethod(_compact_ctor_spy))
    monkeypatch.setattr(
        csv_mod._BENCHMARK_MODULE.CompactBinary,
        'compacy_only_vector',
        classmethod(_compact_only_ctor_spy),
    )

    def _fake_make_simulation_run(**kwargs):
        _record('wrapper.make_simulation_run')
        forwarded_calls.append(kwargs)

        def _run():
            benchmark_mod = csv_mod._BENCHMARK_MODULE
            source_size = 3
            target_size = 5
            conn = benchmark_mod._make_post_conn(
                source_size,
                target_size,
                kwargs['conn_num'],
                efferent_target=kwargs['efferent_target'],
                data_type=kwargs['data_type'],
                homo=kwargs['homo'],
                conn_weight_base=0.6 * u.mS,
                mv_layout=kwargs['mv_layout'],
            )
            spikes = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.bool_)
            result = benchmark_mod._apply_conn(
                spikes,
                conn,
                data_type=kwargs['data_type'],
                efferent_target=kwargs['efferent_target'],
            )
            result_value, _ = u.split_mantissa_unit(result)
            return conn.shape[0], jnp.asarray(result_value).sum()

        return _run

    monkeypatch.setattr(csv_mod, 'make_simulation_run', _fake_make_simulation_run)

    csv_mod.benchmark_conn(
        mode=mode,
        conn_num=20,
        data_type=data_type,
        duration=1 * u.ms,
        homo=True,
        backend='jax_raw',
        params_type='conn',
        mv_layout=mv_layout,
    )

    assert announce_calls == ['called']
    assert backend_calls == [('cpu', 'jax_raw')]
    assert len(forwarded_calls) == 1
    assert forwarded_calls[0]['efferent_target'] == mode
    assert forwarded_calls[0]['data_type'] == data_type
    assert forwarded_calls[0]['mv_layout'] == mv_layout

    assert len(terminal_calls) >= 1
    terminal_name, terminal_kwargs = terminal_calls[0]
    assert terminal_name == expected_terminal
    assert terminal_kwargs['transpose'] is expected_transpose

    has_col_scatter = (
        terminal_kwargs.get('col_weights') is not None
        and terminal_kwargs.get('col_indices') is not None
        and terminal_kwargs.get('col_indptr') is not None
    )
    assert has_col_scatter is expect_col_scatter

    expected_chain = [
        'wrapper.make_simulation_run',
        'benchmark._make_post_conn',
        'benchmark._apply_conn',
        'benchmark._prepare_operand',
    ]
    compact_ctor_name = _expected_compact_ctor_name(data_type, mode)
    if compact_ctor_name is not None:
        expected_chain.append(compact_ctor_name)
    expected_chain.append(
        'main.FixedPostNumConn.__rmatmul__' if mode == 'post'
        else 'main.FixedPostNumConn.__matmul__'
    )
    expected_chain.append(expected_terminal)

    assert call_chain == expected_chain


def test_coba_ei_route_example_call_chain():
    chain = [
        'COBA_EI_binary_fcnmv_CsvOuput.benchmark_conn',
        'COBA_EI_binary_fcnmv_CsvOuput.make_simulation_run',
        'COBA EI benchmark._make_post_conn',
        'COBA EI benchmark._apply_conn',
        'COBA EI benchmark._prepare_operand',
        'FixedPostNumConn.__matmul__',
        'binary_fcnmv',
    ]
    assert chain[-1] == 'binary_fcnmv'


def test_coba_ei_non_repeat_forwards_resume_target_for_bt_generated_pairs(monkeypatch):
    fake_bt = _install_benchmark_tools_stub(monkeypatch)
    fake_bt.filtered_states = [(2, None, 20)]
    csv_mod = _load_coba_ei_csv_module(monkeypatch)

    announce_calls = []
    backend_calls = []
    forwarded_calls = []

    def _announce():
        announce_calls.append('called')
        return 'cpu'

    def _set_backend(platform, backend):
        backend_calls.append((platform, backend))

    def _fake_make_simulation_run(**kwargs):
        forwarded_calls.append(kwargs)

        def _run():
            return 7, jnp.asarray(0.5, dtype=jnp.float32)

        return _run

    monkeypatch.setattr(csv_mod, '_announce_runtime_platform', _announce)
    monkeypatch.setattr(csv_mod.brainevent.config, 'set_backend', _set_backend)
    monkeypatch.setattr(csv_mod, 'make_simulation_run', _fake_make_simulation_run)
    monkeypatch.setattr(csv_mod, '_run_and_block', lambda run: run())
    monkeypatch.setattr(csv_mod, '_release_run', lambda run: None)

    csv_mod.benchmark_conn(
        mode='post',
        conn_num=20,
        data_type='binary',
        duration=1 * u.ms,
        homo=True,
        backend='jax_raw',
        params_type='dist',
        mv_layout='row_gather',
        non_repeat=True,
    )

    assert announce_calls == ['called']
    assert backend_calls == [('cpu', 'jax_raw')]
    assert fake_bt.init_kwargs == [
        {
            'limit_GB': 16,
            '_N': 4000,
            'sample_points': 40,
            'conn_max': 4000,
            'scale_max': 2000,
            'mode': 'post',
            'data_type': 'binary',
            'mv_layout': 'row_gather',
            'non_repeat': True,
            'flush_file_name': str(
                Path(csv_mod.__file__).resolve().parent
                / 'benchmarker-test'
                / 'memorylimittemp-coba-ei_binary_homo_jax_raw_post-row_gather-float-input-16GB.csv'
            ),
        }
    ]
    assert len(fake_bt.generate_calls) == 1
    assert fake_bt.filter_calls == []
    assert len(forwarded_calls) == 1
    assert forwarded_calls[0]['scale'] == 2
    assert forwarded_calls[0]['conn_num'] == 20
