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

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import numpy as np

import brainstate
import brainunit as u

import brainevent._fcn.main as fcn_main_mod
import brainevent._fcn.binary as binary_mod
from dev.fcn.coba_ei_benchmark_test_helpers import (
    coba_ei_module,
    install_fake_brainpy_state,
    run_batch_e2e_spike_history_once,
    run_batch_simulation_once,
)


SCALE = 20
CONN_NUM = 20
BATCH_SIZE = 8
STEPS = 50
SEED = 0
DURATION = 5.0 * u.ms
DATA_TYPE = 'binary'
EFFERENT_TARGET = 'post'
MV_LAYOUT = 'row_gather'
HOMO = True
BACKENDS = (
    'jax_raw',
    'test_colmajor_fullwarp_nocap',
)


def _rate_scalar(rate):
    return float(jax.device_get(u.get_mantissa(rate)).reshape(()))


def _run_rate_fake_state(mod, backend: str):
    result = run_batch_simulation_once(
        mod,
        seed=SEED,
        scale=SCALE,
        batch_size=BATCH_SIZE,
        data_type=DATA_TYPE,
        efferent_target=EFFERENT_TARGET,
        duration=DURATION,
        conn_num=CONN_NUM,
        homo=HOMO,
        mv_layout=MV_LAYOUT,
        backend=backend,
    )
    num, rate = result
    return int(num), _rate_scalar(rate)


def _run_rate_real_state(mod, backend: str):
    brainstate.random.seed(SEED)
    run = mod.make_simulation_batch_run(
        scale=SCALE,
        batch_size=BATCH_SIZE,
        data_type=DATA_TYPE,
        efferent_target=EFFERENT_TARGET,
        duration=DURATION,
        conn_num=CONN_NUM,
        homo=HOMO,
        mv_layout=MV_LAYOUT,
        backend=backend,
    )
    num, rate = jax.block_until_ready(run())
    return int(num), _rate_scalar(rate)


def _count_route_calls(mod, backend: str):
    calls = {'mv': 0, 'mm': 0}
    orig_mv = fcn_main_mod.binary_fcnmv
    orig_mm = fcn_main_mod.binary_fcnmm

    def _wrap_mv(*args, **kwargs):
        calls['mv'] += 1
        return orig_mv(*args, **kwargs)

    def _wrap_mm(*args, **kwargs):
        calls['mm'] += 1
        return orig_mm(*args, **kwargs)

    fcn_main_mod.binary_fcnmv = _wrap_mv
    fcn_main_mod.binary_fcnmm = _wrap_mm
    try:
        _run_rate_real_state(mod, backend)
    finally:
        fcn_main_mod.binary_fcnmv = orig_mv
        fcn_main_mod.binary_fcnmm = orig_mm
    return calls


#===========================
# primitive dispatch debug
#===========================
def _capture_primitive_calls(mod, backend: str):
    calls = []
    orig_mv_p_call = binary_mod.binary_fcnmv_p_call
    orig_mm_p_call = binary_mod.binary_fcnmm_p_call

    def _wrap_mv_p_call(*args, **kwargs):
        spikes = args[2]
        calls.append({
            'primitive': 'binary_fcnmv_p_call',
            'backend': kwargs.get('backend'),
            'spikes_ndim': getattr(spikes, 'ndim', None),
            'spikes_shape': tuple(getattr(spikes, 'shape', ())),
            'transpose': kwargs.get('transpose'),
        })
        return orig_mv_p_call(*args, **kwargs)

    def _wrap_mm_p_call(*args, **kwargs):
        matrix = args[2]
        calls.append({
            'primitive': 'binary_fcnmm_p_call',
            'backend': kwargs.get('backend'),
            'matrix_ndim': getattr(matrix, 'ndim', None),
            'matrix_shape': tuple(getattr(matrix, 'shape', ())),
            'transpose': kwargs.get('transpose'),
        })
        return orig_mm_p_call(*args, **kwargs)

    binary_mod.binary_fcnmv_p_call = _wrap_mv_p_call
    binary_mod.binary_fcnmm_p_call = _wrap_mm_p_call
    try:
        _run_rate_real_state(mod, backend)
    finally:
        binary_mod.binary_fcnmv_p_call = orig_mv_p_call
        binary_mod.binary_fcnmm_p_call = orig_mm_p_call
    return calls


def _compare_spikes(mod, backend: str):
    actual, expected = run_batch_e2e_spike_history_once(
        mod,
        scale=SCALE,
        batch_size=BATCH_SIZE,
        data_type=DATA_TYPE,
        efferent_target=EFFERENT_TARGET,
        conn_num=CONN_NUM,
        homo=HOMO,
        mv_layout=MV_LAYOUT,
        actual_backend=backend,
        reference_backend='jax_raw',
        steps=STEPS,
        conn_seed=SEED,
        state_seed=SEED,
    )
    actual_np = np.asarray(jax.device_get(actual), dtype=np.float32)
    expected_np = np.asarray(jax.device_get(expected), dtype=np.float32)
    same = np.array_equal(actual_np, expected_np)
    diff_count = int(np.count_nonzero(actual_np != expected_np))
    return same, diff_count, actual_np.shape


def main():
    mod = coba_ei_module()
    class _MonkeyPatchShim:
        @staticmethod
        def setattr(obj, name, value):
            setattr(obj, name, value)

    install_fake_brainpy_state(mod, _MonkeyPatchShim())
    print('Runtime platform:', jax.default_backend())
    print(
        f'Debug point: scale={SCALE}, conn_num={CONN_NUM}, '
        f'batch_size={BATCH_SIZE}, duration={DURATION}, steps={STEPS}'
    )
    print()

    print('Real benchmark run rate:')
    for backend in BACKENDS:
        n, rate = _run_rate_real_state(mod, backend)
        print(f'{backend:>28s} | neurons={n:>7d} | rate={rate:>9.6f} Hz')

    print()

    print('Real benchmark route counts:')
    for backend in BACKENDS:
        calls = _count_route_calls(mod, backend)
        print(f'{backend:>28s} | mv_calls={calls["mv"]} | mm_calls={calls["mm"]}')

    print()

    #===========================
    # primitive dispatch report
    #===========================
    print('Primitive dispatch trace:')
    for backend in BACKENDS:
        prim_calls = _capture_primitive_calls(mod, backend)
        print(f'  backend={backend}')
        for call in prim_calls:
            print(f'    {call}')

    print()

    print('Fake-state helper run rate:')
    for backend in BACKENDS:
        n, rate = _run_rate_fake_state(mod, backend)
        print(f'{backend:>28s} | neurons={n:>7d} | rate={rate:>9.6f} Hz')

    print()

    for backend in BACKENDS[1:]:
        brainstate.random.seed(SEED)
        same, diff_count, shape = _compare_spikes(mod, backend)
        print(
            f'spike compare vs jax_raw: {backend:>16s} | '
            f'same={same} | diff_count={diff_count} | shape={shape}'
        )


if __name__ == '__main__':
    main()
