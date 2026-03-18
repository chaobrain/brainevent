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

#
# Benchmark: compact_binary vs bitpack_binary vs binary (COBA 2005)
#
# Compares three event representations for event-driven sparse MM:
#   - compact   (bitpack + stream compaction, fused CUDA kernel)
#   - bitpack   (bitpack only, axis=0)
#   - binary    (unpacked boolean baseline)
#
# CompactBinary advantages:
#   - Scatter mode: only iterates active rows via active_ids (5-15% firing → 85-95% skipped)
#   - Fused CUDA kernel for bitpack + compaction (single kernel launch vs 5+ JAX ops)
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
from COBA_2005_benchmark import make_simulation_batch_run
import CsvOutput as RP

brainevent.config.set_backend('gpu', 'cuda_raw')

backends = ['jax_raw', 'cuda_raw']
scales = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
conn_nums = [20, 40, 80, 160, 320, 640]
probs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]


def benchmark_post_conn(
    conn_num=None, 
    conn_prob=None, 
    data_type='compact', 
    batch_size=16, 
    duration=1e3 * u.ms, 
    homo: bool = False, 
    backend: str | None = None,
    probs_or_conn='conn'
):
    print('Benchmarking post-synaptic connection updates...')

    backends_to_use = [backend] if backend is not None else backends

    if probs_or_conn == 'conn':
        use_conn_nums = True
        conn_nums_to_use = [conn_num] if conn_num is not None else conn_nums
    else:
        use_conn_nums = False
        probs_to_use = [conn_prob] if conn_prob is not None else probs

    csv_recorder = RP.CSV_record(f'compact_post_bs{batch_size}', 'fcnmm', 'coba', duration=duration, conn=conn_num)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)

        if use_conn_nums:
            for cn in conn_nums_to_use:
                csv_recorder.print_header(operator='fcnmm', data_type=data_type, backend=back,
                                          mode='post', batch_size=batch_size, conn_num=cn, duration=duration,
                                          homo=('homo' if homo else 'hetero'))
                csv_recorder.print_table_header()

                for s in scales:
                    try:
                        run = make_simulation_batch_run(
                            scale=s,
                            batch_size=batch_size,
                            data_type=data_type,
                            efferent_target='post',
                            duration=duration,
                            conn_num=cn,
                            homo=homo,
                        )
                        # warmup
                        jax.block_until_ready(run())
                        # timed run
                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        
                        csv_recorder.print_row(s, n, elapsed, float(rate))
                        csv_recorder.single_COBA_data_add(
                            'fcnmm', data_type, back, 'post', cn, s,
                            elapsed, float(rate), duration, homo=('homo' if homo else 'hetero')
                        )
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={cn}: {e}')
                        continue
        else:
            for prob in probs_to_use:
                csv_recorder.print_header(operator='fcnmm', data_type=data_type, backend=back,
                                          mode='post', batch_size=batch_size, duration=duration,
                                          homo=('homo' if homo else 'hetero'), prob=prob)
                csv_recorder.print_table_header(show_conn=True)

                for s in scales:
                    actual_conn_num = s * prob
                    try:
                        run = make_simulation_batch_run(
                            scale=s,
                            batch_size=batch_size,
                            data_type=data_type,
                            efferent_target='post',
                            duration=duration,
                            conn_num=actual_conn_num,
                            homo=homo,
                        )
                        # warmup
                        jax.block_until_ready(run())
                        # timed run
                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        
                        csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=actual_conn_num)
                        csv_recorder.single_COBA_data_add(
                            'fcnmm', data_type, back, 'post', actual_conn_num, s,
                            elapsed, float(rate), duration, homo=('homo' if homo else 'hetero')
                        )
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={actual_conn_num}: {e}')
                        continue
    csv_recorder.record_finish('default')


def benchmark_pre_conn(
    conn_num=None, 
    conn_prob=None,
    data_type='compact', 
    batch_size=16, 
    duration=1e2 * u.ms, 
    homo: bool = False, 
    backend: str | None = None,
    probs_or_conn='conn'
):
    print('Benchmarking pre-synaptic connection updates...')

    backends_to_use = [backend] if backend is not None else backends

    if probs_or_conn == 'conn':
        use_conn_nums = True
        conn_nums_to_use = [conn_num] if conn_num is not None else conn_nums
    else:
        use_conn_nums = False
        probs_to_use = [conn_prob] if conn_prob is not None else probs

    csv_recorder = RP.CSV_record(f'compact_pre_bs{batch_size}', 'fcnmm', 'coba', duration=duration, conn=conn_num)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)

        if use_conn_nums:
            for cn in conn_nums_to_use:
                csv_recorder.print_header(operator='fcnmm', data_type=data_type, backend=back,
                                          mode='scatter', batch_size=batch_size, conn_num=cn, duration=duration,
                                          homo=('homo' if homo else 'hetero'))
                csv_recorder.print_table_header()

                for s in scales:
                    try:
                        run = make_simulation_batch_run(
                            scale=s,
                            batch_size=batch_size,
                            data_type=data_type,
                            efferent_target='pre',
                            duration=duration,
                            conn_num=cn,
                            homo=homo,
                        )
                        # warmup
                        jax.block_until_ready(run())
                        # timed run
                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        
                        csv_recorder.print_row(s, n, elapsed, float(rate))
                        csv_recorder.single_COBA_data_add(
                            'fcnmm', data_type, back, 'pre', cn, s,
                            elapsed, float(rate), duration, homo=('homo' if homo else 'hetero')
                        )
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={cn}: {e}')
                        continue
        else:
            for prob in probs_to_use:
                csv_recorder.print_header(operator='fcnmm', data_type=data_type, backend=back,
                                          mode='scatter', batch_size=batch_size, duration=duration,
                                          homo=('homo' if homo else 'hetero'), prob=prob)
                csv_recorder.print_table_header(show_conn=True)

                for s in scales:
                    actual_conn_num = s * prob  # non-linear: conn_num scales with network size
                    try:
                        run = make_simulation_batch_run(
                            scale=s,
                            batch_size=batch_size,
                            data_type=data_type,
                            efferent_target='pre',
                            duration=duration,
                            conn_num=actual_conn_num,
                            homo=homo,
                        )
                        # warmup
                        jax.block_until_ready(run())
                        # timed run
                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        
                        csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=actual_conn_num)
                        csv_recorder.single_COBA_data_add(
                            'fcnmm', data_type, back, 'pre', actual_conn_num, s,
                            elapsed, float(rate), duration, homo=('homo' if homo else 'hetero')
                        )
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={actual_conn_num}: {e}')
                        continue
    csv_recorder.record_finish('default')


def compare_all(efferent_target='post', batch_size=16, conn_num=80, duration=1e3 * u.ms):
    # This wrapper maintains the original script's behavior of running all types
    # but uses the new benchmark functions.
    data_types = ['binary', 'bitpack_a0', 'compact']
    
    if efferent_target == 'post':
        for dt in data_types:
            benchmark_post_conn(conn_num=conn_num, data_type=dt, batch_size=batch_size, duration=duration)
    else:
        for dt in data_types:
            # Note: pre mode often uses shorter duration in original script
            benchmark_pre_conn(conn_num=conn_num, data_type=dt, batch_size=batch_size, duration=duration)


if __name__ == '__main__':
    print('#' * 90)
    print('# CompactBinary vs BitPackBinary vs Binary — COBA 2005 Benchmark')
    print('#' * 90)

    # Post-synaptic (gather mode)
    compare_all(efferent_target='post', batch_size=16, conn_num=80)

    print('\n\n')

    # Pre-synaptic (scatter mode)
    compare_all(efferent_target='pre', batch_size=16, conn_num=80, duration=1e2 * u.ms)
