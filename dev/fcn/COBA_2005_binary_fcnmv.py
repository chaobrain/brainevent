# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

#
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#


import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time

import brainunit as u
import jax

import brainevent
from COBA_2005_benchmark import make_simulation_run
from CsvOutput import CSV_record, ResultPrinting

#brainevent.config.set_backend('gpu', 'jax_raw')
brainevent.config.set_backend('gpu', 'cuda_raw')


scales = [1, 4,  8,  20,  60, 100]
backends = ['jax_raw', 'cuda_raw', 'cuda_wprNT']


rp = ResultPrinting()
homo = True


def benchmark_post_conn(
    conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cuda_raw',
):
    # --------------------------------
    # 2026/03/08, conn_num, data_type, duration = 80, 'binary', 1e4 * u.ms
    # --------------------------------
    #
    # scale=1, size=4000, time = 1.3123691082000732 s, firing rate = 59.57392883300781 Hz
    # scale=2, size=8000, time = 1.4384987354278564 s, firing rate = 59.57027816772461 Hz
    # scale=4, size=16000, time = 1.5230646133422852 s, firing rate = 59.569297790527344 Hz
    # scale=6, size=24000, time = 1.6051039695739746 s, firing rate = 59.56822967529297 Hz
    # scale=8, size=32000, time = 1.53263258934021 s, firing rate = 59.57051086425781 Hz
    # scale=10, size=40000, time = 1.6150670051574707 s, firing rate = 59.57093811035156 Hz
    # scale=20, size=80000, time = 2.0971567630767822 s, firing rate = 59.56829833984375 Hz
    # scale=40, size=160000, time = 2.7911770343780518 s, firing rate = 59.57014465332031 Hz
    # scale=60, size=240000, time = 4.094450235366821 s, firing rate = 59.57121276855469 Hz
    # scale=80, size=320000, time = 5.091517448425293 s, firing rate = 59.57069396972656 Hz
    # scale=100, size=400000, time = 6.501585245132446 s, firing rate = 59.56977462768555 Hz
    #
    # --------------------------------
    # 2026/03/08, conn_num, data_type, duration = 80, 'bitpack', 1e4 * u.ms
    # --------------------------------
    #
    # scale=1, size=4000, time = 1.3123691082000732 s, firing rate = 59.57392883300781 Hz
    # scale=2, size=8000, time = 1.4384987354278564 s, firing rate = 59.57027816772461 Hz
    # scale=4, size=16000, time = 1.5230646133422852 s, firing rate = 59.569297790527344 Hz
    # scale=6, size=24000, time = 1.6051039695739746 s, firing rate = 59.56822967529297 Hz
    # scale=8, size=32000, time = 1.53263258934021 s, firing rate = 59.57051086425781 Hz
    # scale=10, size=40000, time = 1.6150670051574707 s, firing rate = 59.57093811035156 Hz
    # scale=20, size=80000, time = 2.0971567630767822 s, firing rate = 59.56829833984375 Hz
    # scale=40, size=160000, time = 2.7911770343780518 s, firing rate = 59.57014465332031 Hz
    # scale=60, size=240000, time = 4.094450235366821 s, firing rate = 59.57121276855469 Hz
    # scale=80, size=320000, time = 5.091517448425293 s, firing rate = 59.57069396972656 Hz
    # scale=100, size=400000, time = 6.501585245132446 s, firing rate = 59.56977462768555 Hz


    #
    # --------------------------------
    # 2026/02/13, AMD Ryzen 7 7840HS, brainevent 0.0.6, Numba 0.63.1, jax 0.9.0, 小新Win11野兽模式
    # --------------------------------
    #
    # scale=1, size=4000, time = 7.121581077575684 s, firing rate = 59.563377380371094 Hz
    # scale=2, size=8000, time = 11.442655324935913 s, firing rate = 59.57026672363281 Hz
    # scale=4, size=16000, time = 15.775666236877441 s, firing rate = 59.571495056152344 Hz
    # scale=6, size=24000, time = 19.378953218460083 s, firing rate = 59.57106018066406 Hz
    # scale=8, size=32000, time = 22.951914310455322 s, firing rate = 59.57032775878906 Hz
    # scale=10, size=40000, time = 26.57138180732727 s, firing rate = 59.57281494140625 Hz
    # scale=20, size=80000, time = 39.93744134902954 s, firing rate = 59.56884765625 Hz
    # scale=40, size=160000, time = 72.9319748878479 s, firing rate = 59.570556640625 Hz
    # scale=60, size=240000, time = 105.03494596481323 s, firing rate = 59.57072067260742 Hz
    # scale=80, size=320000, time = 127.52736496925354 s, firing rate = 59.57038879394531 Hz
    # scale=100, size=400000, time = 159.3613338470459 s, firing rate = 59.569969177246094 Hz
    #
    # --------------------------------
    # 2026/03/08, AMD Ryzen 7 7840HS, conn_num, data_type, duration = 80, 'binary', 1e4 * u.ms, jax 0.9.1, 小新Win11野兽模式
    # --------------------------------
    #
    # scale=1, size=4000, time = 7.930975675582886 s, firing rate = 59.55842971801758 Hz
    # scale=2, size=8000, time = 12.520666599273682 s, firing rate = 59.56388854980469 Hz
    # scale=4, size=16000, time = 18.810335874557495 s, firing rate = 59.570133209228516 Hz
    # scale=6, size=24000, time = 24.842906713485718 s, firing rate = 59.56852722167969 Hz
    # scale=8, size=32000, time = 30.43906331062317 s, firing rate = 59.567989349365234 Hz
    # scale=10, size=40000, time = 36.534552812576294 s, firing rate = 59.56911849975586 Hz
    # scale=20, size=80000, time = 51.978811502456665 s, firing rate = 59.569488525390625 Hz
    # scale=40, size=160000, time = 84.97480726242065 s, firing rate = 59.56957244873047 Hz
    # scale=60, size=240000, time = 120.33725643157959 s, firing rate = 59.5693473815918 Hz

    brainevent.config.set_backend('gpu', backend)
    print('Benchmarking post-synaptic connection updates...')
    csv_recorder = CSV_record('binary_post', 'fcnmv', 'coba')
    dur_ms = float(duration / u.ms)
    for backend in backends:
        brainevent.config.set_backend('gpu', backend)
        rp.print_header(operator='fcnmv', data_type=data_type, backend=backend,
                mode='post', conn_num=conn_num, duration_ms=dur_ms,
                homo=('homo' if homo else 'hetero'))
        rp.print_table_header()
        for s in scales:
            run = make_simulation_run(
                scale=s,
                data_type=data_type,
                efferent_target='post',
                duration=duration,
                conn_num=conn_num,
                homo=homo
            )

            jax.block_until_ready(run())

            t0 = time.time()
            n, rate = jax.block_until_ready(run())
            t1 = time.time()
            elapsed = t1 - t0
            rp.print_row(s, n, elapsed, float(rate))
            csv_recorder.single_COBA_data_add('fcnmv', data_type, backend, 'post', conn_num, s, elapsed, float(rate), dur_ms, homo=('homo' if homo else 'hetero'))
    csv_recorder.record_finish('default')

def benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e2 * u.ms):
    print('Benchmarking pre-synaptic connection updates...')

    csv_recorder = CSV_record('binary_pre', 'fcnmv', 'coba')
    dur_ms = float(duration / u.ms)
    for backend in backends:
        brainevent.config.set_backend('gpu', backend)
        rp.print_header(operator='fcnmv', data_type=data_type, backend=backend,
                mode='pre', conn_num=conn_num, duration_ms=dur_ms,
                homo=('homo' if homo else 'hetero'))
        rp.print_table_header()
        for s in scales:
            run = make_simulation_run(
                scale=s,
                data_type=data_type,
                efferent_target='pre',
                duration=duration,
                conn_num=conn_num,
                homo=homo,
            )

            jax.block_until_ready(run())

            t0 = time.time()
            n, rate = jax.block_until_ready(run())
            t1 = time.time()
            elapsed = t1 - t0
            rp.print_row(s, n, elapsed, float(rate))
            csv_recorder.single_COBA_data_add('fcnmv', data_type, backend, 'pre', conn_num, s, elapsed, float(rate), dur_ms, homo=('homo' if homo else 'hetero'))

    csv_recorder.record_finish('temp')


if __name__ == '__main__':
    #benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='jax_raw')
    #benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cuda_raw')
    #benchmark_post_conn(conn_num=80, data_type='bitpack', duration=1e4 * u.ms)
    benchmark_pre_conn(conn_num=80,data_type='binary',duration=1e3 * u.ms,)