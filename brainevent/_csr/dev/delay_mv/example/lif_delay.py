"""Run 10,000 recurrent LIF neurons with random delays owned by delay_tcsr."""

import argparse

import jax.numpy as jnp
import numpy as np

from brainevent._csr.main import CSR
from brainevent._tcsr.delay_mv.delay_main import delay_tcsr


def main() -> None:
    """Run and summarize a sparse LIF simulation using stateful left multiplication.

    Notes
    -----
    Use dimensionless voltage, dt = 1 ms and tau = 10 ms. Incoming weighted
    spikes cause instantaneous voltage jumps; voltage decays between steps.
    A neuron fires at voltage >= 1 and resets to zero, without a refractory
    period. Each output spike vector becomes the next projection input.
    The 10,000-by-10,000 matrix has density 0.01 and uniform integer delays
    from 0 through 8, requiring nine ring slots. Seed 2026 fixes all randomness.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--backend", choices=("jax", "cuda_raw", "cusparse"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    n_neurons, density, max_delay = 10_000, 1e-2, 8
    nnz = int(n_neurons * n_neurons * density)
    rng = np.random.default_rng(2026)
    # 无放回抽取矩阵位置，保证精确密度和无重复边，不创建稠密矩阵。
    positions = rng.choice(n_neurons * n_neurons, size=nnz, replace=False)
    rows, columns = np.divmod(positions, n_neurons)
    edge_delays = rng.integers(0, max_delay + 1, size=nnz, dtype=np.int32)
    order = np.lexsort((edge_delays, rows))  # 先按源行，再按 delay 分段。
    indptr = np.zeros(n_neurons + 1, np.int32)
    np.cumsum(np.bincount(rows, minlength=n_neurons), out=indptr[1:])
    source = CSR((
        np.array([0.2], np.float32),  # 每条边共享权重 0.2。
        columns[order].astype(np.int32),
        indptr,
    ), shape=(n_neurons, n_neurons))
    W = delay_tcsr(source, edge_delays[order],
                   binary_backend=args.backend)
    voltage = jnp.zeros(n_neurons, jnp.float32)
    spikes = jnp.asarray(rng.random(n_neurons) < 0.1)  # 仅初始时刻随机播种。
    decay = jnp.float32(np.exp(-1.0 / 10.0))

    print(f"shape={source.shape}, nnz={nnz}, density={density:g}, delay=0..{max_delay}, seed=2026")
    print(f"backend={W.binary_backend}, buffer_slots={W.buffer_slots}, dt=1 ms, tau=10 ms")
    for step in range(args.steps):
        head_before = int(W.head[0])
        arrivals = jnp.asarray(spikes @ W)  # __rmatmul__ 自行消费、清槽、推进 head。
        voltage = decay * voltage + arrivals
        next_spikes = voltage >= 1.0
        voltage = jnp.where(next_spikes, jnp.float32(0), voltage)
        print(
            f"step={step:02d} head={head_before}->{int(W.head[0])} "
            f"input_spikes={int(jnp.count_nonzero(spikes))} "
            f"arrivals_sum={float(jnp.sum(arrivals)):.3f} "
            f"v_mean={float(jnp.mean(voltage)):.3f} "
            f"output_spikes={int(jnp.count_nonzero(next_spikes))}"
        )
        spikes = next_spikes


if __name__ == "__main__":
    main()
