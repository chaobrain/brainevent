# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
from pathlib import Path

import brainstate
import brainunit as u
import jax
import numpy as np
import pytest

from brainevent._event.compact_binary import CompactBinary

pytest.importorskip('brainpy')


def _load_coba_ei_benchmark_module():
    benchmark_path = Path(__file__).resolve().parents[2] / 'dev' / 'fcn' / 'COBA EI benchmark.py'
    spec = importlib.util.spec_from_file_location('coba_ei_benchmark_test_module', benchmark_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ('efferent_target', 'mv_layout'),
    [('pre', 'col_scatter'), ('post', 'row_gather')],
)
def test_coba_ei_compact_route_uses_full_compaction(monkeypatch, efferent_target, mv_layout):
    benchmark_mod = _load_coba_ei_benchmark_module()

    original_from_array = CompactBinary.from_array.__func__
    calls = {'from_array': 0}

    def _spy_from_array(cls, x, bit_width=32):
        calls['from_array'] += 1
        return original_from_array(cls, x, bit_width=bit_width)

    def _fail_from_array_light(cls, x, bit_width=32):
        raise AssertionError('COBA EI compact benchmark path must not call CompactBinary.from_array_light().')

    monkeypatch.setattr(CompactBinary, 'from_array', classmethod(_spy_from_array))
    monkeypatch.setattr(CompactBinary, 'from_array_light', classmethod(_fail_from_array_light))

    np.random.seed(123)
    brainstate.random.seed(999)
    run = benchmark_mod.make_simulation_run(
        scale=1,
        data_type='compact',
        efferent_target=efferent_target,
        duration=5 * u.ms,
        conn_num=20,
        homo=True,
        mv_layout=mv_layout,
    )
    n, rate = jax.block_until_ready(run())

    assert int(n) > 0
    assert np.isfinite(float(rate))
    assert calls['from_array'] > 0
