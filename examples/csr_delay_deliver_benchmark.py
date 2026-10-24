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

"""Per-step cost of per-edge-delay delivery on a Brunel-like network (#203).

Network: ``N`` neurons, fixed in-degree ``K`` (random sources), heterogeneous
float32 weights, ring length ``L``, delays uniform in ``[1, L)``. Each step of
a ``lax.scan`` delivers the spikes of that step, reads slot ``t % L`` and
clears it. Variants:

- ``deliver/<backend>``: ``csr_delay_deliver`` with that backend;
- ``jax_event``: event-driven pure JAX (compact active rows with a fixed
  capacity, gather their edges, one scatter-add), as in brainpy.state;
- ``none``: no delivery (cost of the scan, ring read and clear only);
- ``scalar``: scalar-delay baseline, ``BinaryArray(spikes) @ CSR`` (default
  backend) written into the ring at ``(t + D) % L``.

Run, e.g.::

    python examples/csr_delay_deliver_benchmark.py
    JAX_PLATFORMS=cpu taskset -c 0 python examples/csr_delay_deliver_benchmark.py --steps 50
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import brainevent
from brainevent import BinaryArray, CSR, csr_delay_deliver, csr_delay_deliver_p


def build_network(n, k, n_slots, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, n * k, dtype=np.int32)
    post = np.repeat(np.arange(n, dtype=np.int32), k)
    order = np.argsort(pre, kind='stable')
    pre, post = pre[order], post[order]
    indptr = np.zeros(n + 1, np.int32)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    weights = rng.normal(0.1, 0.02, n * k).astype(np.float32)
    delays = rng.integers(1, n_slots, n * k, dtype=np.int32)
    return weights, post, indptr, delays


def jax_event_deliver(ring, spikes, w, idx, ipt, dl, t, *, row_cap, max_len):
    """Event-driven pure-JAX reference (brainpy.state style): compact active rows, gather, scatter."""
    n_slots = ring.shape[0]
    rows = jnp.nonzero(spikes, size=row_cap, fill_value=0)[0]
    n_active = jnp.sum(spikes)
    start = ipt[rows]
    length = jnp.where(jnp.arange(row_cap) < n_active, ipt[rows + 1] - start, 0)
    j = jnp.arange(max_len)
    e = start[:, None] + j[None, :]
    valid = j[None, :] < length[:, None]
    e = jnp.where(valid, e, 0)
    slot = (t + dl[e]) % n_slots
    return ring.at[slot, idx[e]].add(jnp.where(valid, w[e], 0.))


def make_runner(variant, n, n_slots, scalar_delay, row_cap=None, max_len=None):
    """Return ``run(net, spike_train)``; the network arrays are jit arguments (no constant folding)."""

    def deliver(net, ring, spikes, t):
        csr, dl = net
        w, idx, ipt = csr.data, csr.indices, csr.indptr
        if variant == 'none':
            return ring
        if variant == 'jax_event':
            return jax_event_deliver(ring, spikes, w, idx, ipt, dl, t, row_cap=row_cap, max_len=max_len)
        if variant == 'scalar':
            y = BinaryArray(spikes) @ csr
            return ring.at[(t + scalar_delay) % n_slots].add(y)
        backend = variant.split('/')[1]
        return csr_delay_deliver(ring, spikes, w, idx, ipt, dl, t, shape=(n, n), backend=backend)

    @jax.jit
    def run(net, spike_train):
        def step(carry, spikes):
            ring, t = carry
            ring = deliver(net, ring, spikes, t)
            slot = t % n_slots
            out = ring[slot].sum()
            # Without the barrier XLA copies the whole ring twice per step to
            # order the slot read before the in-place clear.
            out, ring = jax.lax.optimization_barrier((out, ring))
            ring = ring.at[slot].set(0.)
            return (ring, t + 1), out

        init = (jnp.zeros((n_slots, n), jnp.float32), jnp.int32(0))
        return jax.lax.scan(step, init, spike_train)

    return run


def time_runner(run, net, spike_train, repeats):
    """Return (seconds per step, checksum of the arrivals)."""
    (_, _), arrived = jax.block_until_ready(run(net, spike_train))  # compile + warm up
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(run(net, spike_train))
        best = min(best, time.perf_counter() - t0)
    return best / spike_train.shape[0], float(jnp.sum(arrived))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=12_500)
    parser.add_argument('--k', type=int, default=1250)
    parser.add_argument('--slots', type=int, default=200)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rates', type=float, nargs='+', default=[0.01, 0.05])
    parser.add_argument('--variants', nargs='+', default=None)
    args = parser.parse_args()

    platform = jax.default_backend()
    if args.variants is None:
        variants = [f'deliver/{b}' for b in csr_delay_deliver_p.available_backends(platform)] + ['jax_event', 'scalar', 'none']
    else:
        variants = args.variants

    weights, indices, indptr, delays = build_network(args.n, args.k, args.slots)
    net = (CSR((jnp.asarray(weights), jnp.asarray(indices), jnp.asarray(indptr)), shape=(args.n, args.n)),
           jnp.asarray(delays))
    print(f'# brainevent {brainevent.__version__}, jax {jax.__version__}, platform {platform}, '
          f'N={args.n}, K={args.k}, nse={indices.size}, L={args.slots}, steps={args.steps}')
    print('| rate | ' + ' | '.join(variants) + ' |')
    print('|---|' + '---|' * len(variants))
    rng = np.random.default_rng(1)
    for rate in args.rates:
        spike_train = jnp.asarray(rng.random((args.steps, args.n)) < rate)
        times, sums = [], []
        for variant in variants:
            run = make_runner(variant, args.n, args.slots, scalar_delay=args.slots // 2,
                              row_cap=int(spike_train.sum(axis=1).max()), max_len=int(np.diff(indptr).max()))
            per_step, checksum = time_runner(run, net, spike_train, args.repeats)
            times.append(per_step)
            sums.append(checksum)
        print(f'| {rate:.0%} | ' + ' | '.join(f'{t * 1e3:.3f} ms' for t in times) + ' |', flush=True)
        print('#   arrival checksums: ' + ', '.join(f'{v}={c:.4g}' for v, c in zip(variants, sums)), flush=True)


if __name__ == '__main__':
    main()
