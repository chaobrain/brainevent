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
import sys
import types
from pathlib import Path

import brainunit as u
import jax.numpy as jnp
import pytest


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


def _load_coba_2005_benchmark_module(monkeypatch):
    _install_brainpy_stub(monkeypatch)
    benchmark_path = Path(__file__).resolve().parents[2] / 'dev' / 'fcn' / 'COBA_2005_benchmark.py'
    spec = importlib.util.spec_from_file_location('coba_2005_benchmark_test_module', benchmark_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_coba_2005_csv_module(monkeypatch, make_calls):
    fake_benchmark = types.ModuleType('COBA_2005_benchmark')

    def _fake_make_simulation_run(**kwargs):
        make_calls.append(kwargs)

        def _run():
            return 7, jnp.asarray(0.5, dtype=jnp.float32)

        return _run

    fake_benchmark.make_simulation_run = _fake_make_simulation_run
    monkeypatch.setitem(sys.modules, 'COBA_2005_benchmark', fake_benchmark)

    fake_bt = types.ModuleType('BenchmarkTools')

    class _FakeTestingParamsGeneratorMV:
        def __init__(self, *args, **kwargs):
            pass

        def make_simulation_params_probs(self, *args, **kwargs):
            return []

        def generate_params(self, *args, **kwargs):
            return [(1, None, 20)]

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

    wrapper_path = Path(__file__).resolve().parents[2] / 'dev' / 'fcn' / 'COBA_2005_binary_fcnmv_CsvOuput.py'
    spec = importlib.util.spec_from_file_location('coba_2005_csv_test_module', wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ('efferent_target', 'mv_layout', 'expected_ctor', 'expect_col_scatter', 'expected_transpose'),
    [
        ('post', 'row_gather', 'full', False, True),
        ('post', 'col_scatter', 'full', False, True),
        ('pre', 'row_gather', 'light', False, False),
        ('pre', 'auto', 'light', False, False),
        ('pre', 'col_scatter', 'full', True, False),
    ],
)
def test_coba_2005_compact_mv_route_semantics(
    monkeypatch,
    efferent_target,
    mv_layout,
    expected_ctor,
    expect_col_scatter,
    expected_transpose,
):
    benchmark_mod = _load_coba_2005_benchmark_module(monkeypatch)

    ctor_calls = []
    kernel_calls = []

    class _FakeCompact:
        def __init__(self, value):
            self.packed = jnp.zeros((1,), dtype=jnp.uint32)
            self.active_ids = jnp.zeros((value.shape[0],), dtype=jnp.int32)
            self.n_active = jnp.asarray([0], dtype=jnp.int32)
            self.value = value

    def _full_ctor(cls, x, bit_width=32):
        ctor_calls.append('full')
        return _FakeCompact(x)

    def _light_ctor(cls, x, bit_width=32):
        ctor_calls.append('light')
        return _FakeCompact(x)

    def _spy_compact_binary_fcnmv(weights, indices, packed, active_ids, n_active, spikes, **kwargs):
        kernel_calls.append(kwargs)
        out_len = kwargs['shape'][1] if kwargs['transpose'] else kwargs['shape'][0]
        return jnp.zeros((out_len,), dtype=jnp.float32)

    monkeypatch.setattr(benchmark_mod.brainevent.CompactBinary, 'from_array', classmethod(_full_ctor))
    monkeypatch.setattr(benchmark_mod.brainevent.CompactBinary, 'from_array_light', classmethod(_light_ctor))
    monkeypatch.setattr(benchmark_mod.brainevent, 'compact_binary_fcnmv', _spy_compact_binary_fcnmv)

    conn = benchmark_mod.FixedNumConn(
        (4,),
        (6,),
        conn_num=2,
        efferent_target=efferent_target,
        data_type='compact',
        homo=True,
        mv_layout=mv_layout,
    )
    spikes = jnp.asarray([1.0, 0.0, 0.5, 1.0], dtype=jnp.float32)

    conn.update(spikes)

    assert ctor_calls == [expected_ctor]
    assert len(kernel_calls) == 1
    assert kernel_calls[0]['transpose'] is expected_transpose

    if expect_col_scatter:
        assert kernel_calls[0]['col_weights'] is conn.col_weight
        assert kernel_calls[0]['col_indices'] is conn.col_indices
        assert kernel_calls[0]['col_indptr'] is conn.col_indptr
        assert conn.col_weight is not None
        assert conn.col_indices is not None
        assert conn.col_indptr is not None
    else:
        assert kernel_calls[0]['col_weights'] is None
        assert kernel_calls[0]['col_indices'] is None
        assert kernel_calls[0]['col_indptr'] is None


def test_coba_2005_csv_wrapper_forwards_mv_layout_and_uses_runtime_platform(monkeypatch):
    make_calls = []
    csv_mod = _load_coba_2005_csv_module(monkeypatch, make_calls)

    announce_calls = []
    backend_calls = []

    def _announce():
        announce_calls.append('called')
        return 'cpu'

    def _set_backend(platform, backend):
        backend_calls.append((platform, backend))

    monkeypatch.setattr(csv_mod, '_announce_runtime_platform', _announce)
    monkeypatch.setattr(csv_mod.brainevent.config, 'set_backend', _set_backend)

    csv_mod.benchmark_conn(
        mode='pre',
        conn_num=20,
        data_type='compact',
        duration=1 * u.ms,
        homo=True,
        backend='jax_raw',
        params_type='conn',
        mv_layout='col_scatter',
    )

    assert announce_calls == ['called']
    assert backend_calls == [('cpu', 'jax_raw')]
    assert len(make_calls) == len(csv_mod.scales)
    assert all(call['efferent_target'] == 'pre' for call in make_calls)
    assert all(call['data_type'] == 'compact' for call in make_calls)
    assert all(call['mv_layout'] == 'col_scatter' for call in make_calls)
