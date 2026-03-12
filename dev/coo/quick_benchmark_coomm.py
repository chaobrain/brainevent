"""Quick benchmark for binary_coomm optimization verification."""

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

scales = [1, 4, 10, 40, 100]
batch_size = 16
duration = 1e3 * u.ms


def benchmark(efferent_target, backend):
    brainevent.config.set_backend('gpu', backend)
    label = 'post' if efferent_target == 'post' else 'pre'
    print(f'\n--- {label}-synaptic (batch={batch_size}), backend={backend} ---')

    for s in scales:
        dur = duration if efferent_target == 'post' else 1e2 * u.ms
        run = make_simulation_batch_run(
            scale=s,
            batch_size=batch_size,
            data_type='binary',
            efferent_target=efferent_target,
            duration=dur,
            conn_num=80,
        )
        # warmup
        jax.block_until_ready(run())

        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        print(f'  scale={s:3d}, size={n:6d}, time={t1 - t0:8.3f}s, rate={rate:.1f}Hz')


if __name__ == '__main__':
    # Post-synaptic = transpose=True (scatter path)
    benchmark('post', 'cuda_raw')
    benchmark('post', 'warp')

    # Pre-synaptic = transpose=False (gather path)
    benchmark('pre', 'cuda_raw')
    benchmark('pre', 'warp')
