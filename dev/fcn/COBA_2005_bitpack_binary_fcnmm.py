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
# Benchmark: bitpack_binary_fcnmm vs binary_fcnmm
#
# Compares event-driven sparse matrix-matrix multiply with pre-packed binary
# spike matrices (bitpack) against unpacked boolean (binary) using the COBA 2005
# network model.
#
# Tests:
#   - bitpack_a0 (pack along row/source dimension, axis=0)
#   - bitpack_a1 (pack along batch dimension, axis=1)
#   - binary     (baseline, unpacked boolean)
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
from CsvOutput import CSV_record, ResultPrinting

brainevent.config.set_backend('gpu', 'cuda_raw')

backends = ['jax_raw', 'cuda_raw']
rp = ResultPrinting()
homo = False


def benchmark_post_conn(data_type, batch_size=16, conn_num=80, duration=1e3 * u.ms):
    dur_ms = float(duration / u.ms)
    csv_recorder = CSV_record(f'bitpack_post_bs{batch_size}_conn{conn_num}', 'fcnmm', 'coba')

    for backend in backends:
        brainevent.config.set_backend('gpu', backend)
        rp.print_header(operator='fcnmm', data_type=data_type, backend=backend,
                mode='post', batch_size=batch_size, conn_num=conn_num,
                duration_ms=dur_ms, homo=('homo' if homo else 'hetero'))
        rp.print_table_header()

        for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
            run = make_simulation_batch_run(
                scale=s,
                batch_size=batch_size,
                data_type=data_type,
                efferent_target='post',
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
            csv_recorder.single_COBA_data_add('fcnmm', data_type, backend, 'post', conn_num, s, elapsed, float(rate), dur_ms, homo=('homo' if homo else 'hetero'))
    csv_recorder.record_finish('default')


def benchmark_pre_conn(data_type, batch_size=16, conn_num=80, duration=1e2 * u.ms):
    dur_ms = float(duration / u.ms)
    csv_recorder = CSV_record(f'bitpack_pre_bs{batch_size}_conn{conn_num}', 'fcnmm', 'coba')

    for backend in backends:
        brainevent.config.set_backend('gpu', backend)
        rp.print_header(operator='fcnmm', data_type=data_type, backend=backend,
                mode='pre', batch_size=batch_size, conn_num=conn_num,
                duration_ms=dur_ms, homo=('homo' if homo else 'hetero'))
        rp.print_table_header()

        for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
            run = make_simulation_batch_run(
                scale=s,
                batch_size=batch_size,
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
            csv_recorder.single_COBA_data_add('fcnmm', data_type, backend, 'pre', conn_num, s, elapsed, float(rate), dur_ms, homo=('homo' if homo else 'hetero'))
    csv_recorder.record_finish('default')


def compare_bitpack_vs_binary():
    """Compare bitpack vs binary for post-synaptic and pre-synaptic MM.

    Note: Under vmap (brainstate.nn.Map), each batch element is 1D, so
    BitPackedBinary only has packed[0]. The batching rule promotes MV→MM
    with pack_axis=0 automatically. bitpack_a0 and bitpack_a1 are
    equivalent in this scenario; use bitpack_a0 for the vmapped benchmark.

    Results (2026-03-09, RTX 4090, batch_size=16, conn_num=80, duration=1000ms):

    ---- Post-synaptic (gather mode) ----
    Scale | Neurons | Binary   | Bitpack  | Speedup
    1     |   4K    | 0.189s   | 0.210s   | 0.90x
    2     |   8K    | 0.282s   | 0.273s   | 1.03x
    4     |  16K    | 0.564s   | 0.531s   | 1.06x
    6     |  24K    | 0.851s   | 0.805s   | 1.06x
    8     |  32K    | 1.104s   | 1.000s   | 1.10x
    10    |  40K    | 1.446s   | 1.241s   | 1.17x
    20    |  80K    | 3.393s   | 3.073s   | 1.10x
    40    | 160K    | 7.368s   | 6.783s   | 1.09x
    60    | 240K    | 11.601s  | 10.822s  | 1.07x
    80    | 320K    | 16.005s  | 14.955s  | 1.07x
    100   | 400K    | 20.530s  | 19.210s  | 1.07x

    ---- Pre-synaptic (scatter mode, duration=100ms) ----
    Scale | Neurons | Binary   | Bitpack  | Speedup
    1     |   4K    | 0.115s   | 0.139s   | 0.83x
    2     |   8K    | 0.194s   | 0.151s   | 1.28x
    4     |  16K    | 0.384s   | 0.277s   | 1.39x
    6     |  24K    | 0.492s   | 0.408s   | 1.21x
    8     |  32K    | 0.677s   | 0.537s   | 1.26x
    10    |  40K    | 0.856s   | 0.681s   | 1.26x
    20    |  80K    | 1.749s   | 1.438s   | 1.22x
    40    | 160K    | 3.014s   | 2.893s   | 1.04x
    60    | 240K    | 4.662s   | 4.465s   | 1.04x
    80    | 320K    | 6.378s   | 6.275s   | 1.02x
    100   | 400K    | 8.143s   | 8.158s   | 1.00x
    """
    print('#' * 70)
    print('# Bitpack vs Binary FCNMM Benchmark (post-synaptic, gather mode)')
    print('#' * 70)

    batch_size = 16
    conn_num = 80

    # Binary baseline
    benchmark_post_conn('binary', batch_size=batch_size, conn_num=conn_num)

    # Bitpack (pack_axis=0 via batching rule MV→MM promotion)
    benchmark_post_conn('bitpack_a0', batch_size=batch_size, conn_num=conn_num)

    print('\n\n')
    print('#' * 70)
    print('# Bitpack vs Binary FCNMM Benchmark (pre-synaptic, scatter mode)')
    print('#' * 70)

    # Binary baseline
    benchmark_pre_conn('binary', batch_size=batch_size, conn_num=conn_num)

    # Bitpack
    benchmark_pre_conn('bitpack_a0', batch_size=batch_size, conn_num=conn_num)


def compare_different_configs():
    """Compare bitpack vs binary with different batch sizes and conn nums."""

    #
    # ======================================================================
    #   data_type=binary, batch_size=16, conn_num=16
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.156s, rate=53.5 Hz
    #   scale=  2, size=  8000, time=   0.185s, rate=53.5 Hz
    #   scale=  4, size= 16000, time=   0.458s, rate=53.5 Hz
    #   scale=  6, size= 24000, time=   0.660s, rate=53.5 Hz
    #   scale=  8, size= 32000, time=   0.757s, rate=53.5 Hz
    #   scale= 10, size= 40000, time=   0.952s, rate=53.5 Hz
    #   scale= 20, size= 80000, time=   2.092s, rate=53.5 Hz
    #   scale= 40, size=160000, time=   4.230s, rate=53.5 Hz
    #   scale= 60, size=240000, time=   6.426s, rate=53.5 Hz
    #   scale= 80, size=320000, time=   8.638s, rate=53.5 Hz
    #   scale=100, size=400000, time=  10.875s, rate=53.5 Hz
    #
    # ======================================================================
    #   data_type=bitpack_a0, batch_size=16, conn_num=16
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.205s, rate=53.5 Hz
    #   scale=  2, size=  8000, time=   0.204s, rate=53.5 Hz
    #   scale=  4, size= 16000, time=   0.397s, rate=53.5 Hz
    #   scale=  6, size= 24000, time=   0.592s, rate=53.5 Hz
    #   scale=  8, size= 32000, time=   0.735s, rate=53.5 Hz
    #   scale= 10, size= 40000, time=   0.884s, rate=53.5 Hz
    #   scale= 20, size= 80000, time=   1.929s, rate=53.5 Hz
    #   scale= 40, size=160000, time=   3.883s, rate=53.5 Hz
    #   scale= 60, size=240000, time=   5.904s, rate=53.5 Hz
    #   scale= 80, size=320000, time=   7.937s, rate=53.5 Hz
    #   scale=100, size=400000, time=   9.991s, rate=53.5 Hz
    #
    # ======================================================================
    #   data_type=binary, batch_size=16, conn_num=32
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.205s, rate=54.2 Hz
    #   scale=  2, size=  8000, time=   0.225s, rate=54.2 Hz
    #   scale=  4, size= 16000, time=   0.458s, rate=54.2 Hz
    #   scale=  6, size= 24000, time=   0.692s, rate=54.2 Hz
    #   scale=  8, size= 32000, time=   0.816s, rate=54.2 Hz
    #   scale= 10, size= 40000, time=   1.021s, rate=54.2 Hz
    #   scale= 20, size= 80000, time=   2.259s, rate=54.2 Hz
    #   scale= 40, size=160000, time=   4.770s, rate=54.2 Hz
    #   scale= 60, size=240000, time=   7.414s, rate=54.2 Hz
    #   scale= 80, size=320000, time=  10.083s, rate=54.2 Hz
    #   scale=100, size=400000, time=  12.793s, rate=54.2 Hz
    #
    # ======================================================================
    #   data_type=bitpack_a0, batch_size=16, conn_num=32
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.211s, rate=54.2 Hz
    #   scale=  2, size=  8000, time=   0.221s, rate=54.2 Hz
    #   scale=  4, size= 16000, time=   0.454s, rate=54.2 Hz
    #   scale=  6, size= 24000, time=   0.589s, rate=54.2 Hz
    #   scale=  8, size= 32000, time=   0.787s, rate=54.2 Hz
    #   scale= 10, size= 40000, time=   0.953s, rate=54.2 Hz
    #   scale= 20, size= 80000, time=   2.067s, rate=54.2 Hz
    #   scale= 40, size=160000, time=   4.371s, rate=54.2 Hz
    #   scale= 60, size=240000, time=   6.809s, rate=54.2 Hz
    #   scale= 80, size=320000, time=   9.285s, rate=54.2 Hz
    #   scale=100, size=400000, time=  11.799s, rate=54.2 Hz
    #
    # ======================================================================
    #   data_type=binary, batch_size=16, conn_num=80
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.214s, rate=59.5 Hz
    #   scale=  2, size=  8000, time=   0.270s, rate=59.5 Hz
    #   scale=  4, size= 16000, time=   0.551s, rate=59.4 Hz
    #   scale=  6, size= 24000, time=   0.843s, rate=59.4 Hz
    #   scale=  8, size= 32000, time=   1.110s, rate=59.4 Hz
    #   scale= 10, size= 40000, time=   1.446s, rate=59.4 Hz
    #   scale= 20, size= 80000, time=   3.413s, rate=59.4 Hz
    #   scale= 40, size=160000, time=   7.404s, rate=59.4 Hz
    #   scale= 60, size=240000, time=  11.724s, rate=59.4 Hz
    #   scale= 80, size=320000, time=  16.132s, rate=59.4 Hz
    #   scale=100, size=400000, time=  20.597s, rate=59.4 Hz
    #
    # ======================================================================
    #   data_type=bitpack_a0, batch_size=16, conn_num=80
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.207s, rate=59.4 Hz
    #   scale=  2, size=  8000, time=   0.272s, rate=59.5 Hz
    #   scale=  4, size= 16000, time=   0.518s, rate=59.4 Hz
    #   scale=  6, size= 24000, time=   0.776s, rate=59.5 Hz
    #   scale=  8, size= 32000, time=   0.984s, rate=59.4 Hz
    #   scale= 10, size= 40000, time=   1.240s, rate=59.4 Hz
    #   scale= 20, size= 80000, time=   3.089s, rate=59.4 Hz
    #   scale= 40, size=160000, time=   6.798s, rate=59.4 Hz
    #   scale= 60, size=240000, time=  10.867s, rate=59.4 Hz
    #   scale= 80, size=320000, time=  14.979s, rate=59.4 Hz
    #   scale=100, size=400000, time=  19.208s, rate=59.4 Hz
    #
    # ======================================================================
    #   data_type=binary, batch_size=32, conn_num=80
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.229s, rate=59.4 Hz
    #   scale=  2, size=  8000, time=   0.485s, rate=59.4 Hz
    #   scale=  4, size= 16000, time=   0.943s, rate=59.4 Hz
    #   scale=  6, size= 24000, time=   1.483s, rate=59.4 Hz
    #   scale=  8, size= 32000, time=   2.107s, rate=59.4 Hz
    #   scale= 10, size= 40000, time=   2.890s, rate=59.4 Hz
    #   scale= 20, size= 80000, time=   6.478s, rate=59.4 Hz
    #   scale= 40, size=160000, time=  14.455s, rate=59.4 Hz
    #   scale= 60, size=240000, time=  22.674s, rate=59.4 Hz
    #   scale= 80, size=320000, time=  31.008s, rate=59.4 Hz
    #   scale=100, size=400000, time=  39.295s, rate=59.4 Hz
    #
    # ======================================================================
    #   data_type=bitpack_a0, batch_size=32, conn_num=80
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.217s, rate=59.4 Hz
    #   scale=  2, size=  8000, time=   0.439s, rate=59.4 Hz
    #   scale=  4, size= 16000, time=   0.877s, rate=59.4 Hz
    #   scale=  6, size= 24000, time=   1.310s, rate=59.4 Hz
    #   scale=  8, size= 32000, time=   1.892s, rate=59.4 Hz
    #   scale= 10, size= 40000, time=   2.648s, rate=59.4 Hz
    #   scale= 20, size= 80000, time=   6.055s, rate=59.4 Hz
    #   scale= 40, size=160000, time=  13.684s, rate=59.4 Hz
    #   scale= 60, size=240000, time=  21.513s, rate=59.4 Hz
    #   scale= 80, size=320000, time=  29.464s, rate=59.4 Hz
    #   scale=100, size=400000, time=  37.470s, rate=59.4 Hz
    #
    # ======================================================================
    #   data_type=binary, batch_size=16, conn_num=128
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.217s, rate=70.6 Hz
    #   scale=  2, size=  8000, time=   0.281s, rate=70.6 Hz
    #   scale=  4, size= 16000, time=   0.566s, rate=70.6 Hz
    #   scale=  6, size= 24000, time=   0.870s, rate=70.6 Hz
    #   scale=  8, size= 32000, time=   1.147s, rate=70.6 Hz
    #   scale= 10, size= 40000, time=   1.503s, rate=70.6 Hz
    #   scale= 20, size= 80000, time=   3.894s, rate=70.6 Hz
    #   scale= 40, size=160000, time=   9.384s, rate=70.6 Hz
    #   scale= 60, size=240000, time=  15.580s, rate=70.6 Hz
    #   scale= 80, size=320000, time=  22.002s, rate=70.6 Hz
    #   scale=100, size=400000, time=  28.596s, rate=70.6 Hz
    #
    # ======================================================================
    #   data_type=bitpack_a0, batch_size=16, conn_num=128
    # ======================================================================
    #   scale=  1, size=  4000, time=   0.220s, rate=70.6 Hz
    #   scale=  2, size=  8000, time=   0.282s, rate=70.6 Hz
    #   scale=  4, size= 16000, time=   0.531s, rate=70.6 Hz
    #   scale=  6, size= 24000, time=   0.807s, rate=70.6 Hz
    #   scale=  8, size= 32000, time=   1.039s, rate=70.6 Hz
    #   scale= 10, size= 40000, time=   1.300s, rate=70.6 Hz
    #   scale= 20, size= 80000, time=   3.570s, rate=70.6 Hz
    #   scale= 40, size=160000, time=   8.785s, rate=70.6 Hz
    #   scale= 60, size=240000, time=  14.766s, rate=70.6 Hz
    #   scale= 80, size=320000, time=  20.946s, rate=70.6 Hz
    #   scale=100, size=400000, time=  27.220s, rate=70.6 Hz

    print('#' * 70)
    print('# Extended Benchmark: Different batch sizes and conn nums')
    print('#' * 70)

    configs = [
        (16, 16),   # warp kernel path
        (16, 32),   # warp boundary
        (16, 80),   # basic kernel, default
        (32, 80),   # basic, larger batch
        (16, 128),  # basic, more connections
    ]

    for batch_size, conn_num in configs:
        for data_type in ['binary', 'bitpack_a0']:
            benchmark_post_conn(data_type, batch_size=batch_size, conn_num=conn_num)


if __name__ == '__main__':
    # compare_bitpack_vs_binary()
    compare_different_configs()
