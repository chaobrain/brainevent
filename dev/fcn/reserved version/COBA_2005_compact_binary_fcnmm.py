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

brainevent.config.set_backend('gpu', 'cuda_raw')


def benchmark_one(data_type, efferent_target, scales, batch_size=16, conn_num=80, duration=1e3 * u.ms):
    results = {}
    for s in scales:
        run = make_simulation_batch_run(
            scale=s,
            batch_size=batch_size,
            data_type=data_type,
            efferent_target=efferent_target,
            duration=duration,
            conn_num=conn_num,
        )
        # warmup
        jax.block_until_ready(run())
        # timed run
        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        results[s] = (n, t1 - t0, float(rate))
    return results


def compare_all(efferent_target='post', batch_size=16, conn_num=80, duration=1e3 * u.ms):
    mode = 'gather' if efferent_target == 'post' else 'scatter'
    scales = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]

    data_types = ['binary', 'bitpack_a0', 'compact']
    all_results = {}
    for dt in data_types:
        print(f'Running {dt} ({mode})...')
        all_results[dt] = benchmark_one(
            dt, efferent_target, scales,
            batch_size=batch_size, conn_num=conn_num, duration=duration,
        )

    # Print comparison table
    print(f'\n{"=" * 90}')
    print(f'  {mode.upper()} mode | batch_size={batch_size}, conn_num={conn_num}, duration={duration}')
    print(f'{"=" * 90}')
    header = f'{"Scale":>5s} | {"Neurons":>7s}'
    for dt in data_types:
        header += f' | {dt:>10s}'
    header += ' | compact vs binary | compact vs bitpack'
    print(header)
    print('-' * len(header))

    for s in scales:
        n = all_results['binary'][s][0]
        row = f'{s:>5d} | {n:>7d}'
        times = {}
        for dt in data_types:
            t = all_results[dt][s][1]
            times[dt] = t
            row += f' | {t:>9.3f}s'
        speedup_vs_binary = times['binary'] / times['compact'] if times['compact'] > 0 else float('inf')
        speedup_vs_bitpack = times['bitpack_a0'] / times['compact'] if times['compact'] > 0 else float('inf')
        row += f' |             {speedup_vs_binary:>5.2f}x |              {speedup_vs_bitpack:>5.2f}x'
        print(row)


if __name__ == '__main__':
    print('#' * 90)
    print('# CompactBinary vs BitPackBinary vs Binary — COBA 2005 Benchmark')
    print('#' * 90)

    # Post-synaptic (gather mode) — compact should match bitpack
    compare_all(efferent_target='post', batch_size=16, conn_num=80)

    print('\n\n')

    # Pre-synaptic (scatter mode) — compact should be faster (skips inactive rows)
    compare_all(efferent_target='pre', batch_size=16, conn_num=80, duration=1e2 * u.ms)
