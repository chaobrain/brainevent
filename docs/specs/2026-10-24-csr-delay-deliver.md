# Fused per-edge-delay spike delivery (`csr_delay_deliver`)

Issue: #203. Branch: `feat/csr-delay-deliver`.

Status: implemented

## Problem

Networks with per-synapse transmission delays need, every step, to add the
weight of every edge of every spiking source into a post-side *arrival ring*
at the slot `(t + d_e) mod L`. brainpy.state does this in pure JAX (gather of
active rows, compaction, segment gather, scatter-add, overflow `lax.cond`).
At Brunel order 2500 (12 500 neurons, 15.6 M edges, L = 200) that is
1.4-1.8x slower than the scalar-delay step (`BinaryArray @ CSR`): random
writes into a 10 MB ring on CPU, ~10 small kernels and a host sync on GPU.
A single fused event-driven kernel removes both costs.

## API

```python
brainevent.csr_delay_deliver(
    ring, events, data, indices, indptr, delays, t,
    *, shape, weighted=False, backend=None,
) -> ring_new

CSR.delay_deliver(ring, events, delays, t, *, weighted=False, backend=None)
```

| argument | shape / dtype | notes |
|---|---|---|
| `ring` | `(L, n_post)`, float | arrival buffer; its dtype is the output dtype |
| `events` | `(n_pre,)` | raw array or `BinaryArray` (unwrapped) |
| `data` | `(1,)`/scalar (homogeneous) or `(nse,)` | cast to `ring.dtype` |
| `indices`, `indptr` | CSR structure of the `(n_pre, n_post)` matrix | int32 indices, int32/int64 indptr |
| `delays` | `(nse,)` integer, or a scalar broadcast to `(nse,)` | steps |
| `t` | integer scalar | may be traced (e.g. inside `lax.scan`); passed as an array operand |
| `shape` | `(n_pre, n_post)` | static |
| `weighted` | static bool | `False`: binary; `True`: event value weighted |

### Semantics

For every edge `e` of row `i = row(e)`:

```
ring_new[(t + d_e) mod L, indices[e]] += w_e * f(x)[i]
f(x)[i] = event_active(x[i])   (binary;   bool: x, numeric: x != 0)
f(x)[i] = x[i]                 (weighted; inactive x[i] == 0 rows are skipped)
```

`mod` is the floor modulo (result in `[0, L)`) on every backend, so a traced
`t` may grow without bound and wraps correctly; `t` and `d` are reduced
modulo `L` separately to avoid int32 overflow of `t + d`. Collisions (several
edges hitting the same `(slot, post)`) accumulate.

Delay validation: when `delays` is concrete, `1 <= d < L` is required and a
`ValueError` is raised otherwise (`d = 0` would write into the slot being read
this step, `d >= L` would alias a later step). When `delays` is traced no check
is possible; out-of-range values wrap modulo `L` with the same formula on every
backend.

Weighted mode is the internal path a forthcoming `WeightedArray` event type
will dispatch to; it is reachable today via `weighted=True`.

### Units

Like `csrmv`: `data` and `events` are split into mantissa and unit; the ring
must carry `unit(data) * unit(events)` (converted with `to_decimal`; a
dimension mismatch raises). The result carries the same unit.

## Primitive `csr_delay_deliver_p`

Operands `(ring, events, data, indices, indptr, delays, t)`, one output
(`ring_new`, aliased to `ring` on the in-place backends). Static params:
`shape`, `weighted`, `backend` and the `*_info` structs.

Backends:

- `jax_raw` (cpu/gpu/tpu): `row_ids = repeat(arange(m), diff(indptr))`,
  one scatter-add `ring.at[slot, indices].add(w * where(active, f(x)[row], 0))`.
  Not event-driven; used as the reference and AD backend.
- `numba` (cpu): serial row loop that skips inactive rows, in-place via
  `input_output_aliases={0: 0}` (rows collide on posts, so no `prange`).
  Each active row is done in two passes (flat ring offsets into a scratch
  buffer, then the scatter), which decouples index loads from the random
  read-modify-writes (~1.7x faster on a 10 MB ring). float32/float64 rings
  only (numba has no float16/bfloat16); other dtypes raise a clear error.
- `cuda_raw` (gpu, default): `csr_delay_deliver.cu`. Thread-, warp- or
  block-per-row chosen by `avg_nnz = nse / n_pre` (`< 32`, `< 256`, else),
  warp-ballot early exit for inactive rows, atomic adds into the aliased
  ring. Entry points `csr_delay_deliver{_f32,_f64,_f16,_bf16}{_bool,_float,_weighted}`;
  homogeneous vs heterogeneous weights and int32/int64 indptr are dispatched
  at runtime inside each entry. Binary events use `cuda_event_abi`
  (`prepare` applied inside the lowering); weighted events are cast to the
  weight dtype in the lowering; delays and `t` are cast to int32 there.

## Autodiff

The output is linear in `ring` and `data`, and (weighted mode) in `events`.
With `s_e = (t + d_e) mod L` and `ct` the output cotangent:

- `d ring = ct`;
- `d data[e] = f(x)[row(e)] * ct[s_e, indices[e]]` (homogeneous: summed);
- `d x[i] = sum_{e in row i} w_e * ct[s_e, indices[e]]` (weighted: exact;
  binary: straight-through estimator, same formula);
- `delays`, `t`, `indices`, `indptr`: non-differentiable.

JVP: ring tangent passes through; data tangent is the primitive on a zero
ring with `data_dot`; event tangent is the *weighted* primitive on a zero
ring with `x_dot` (this is the straight-through rule in binary mode). The
transpose rule implements the cotangents above with gathers and a
`segment_sum` in JAX.

## Batching

`vmap` over trials (batched `ring` and/or `events`, or anything else) uses
`general_batching_rule` (a `lax.scan` over the batch). A fused batched
kernel is future work.

## Tests (`brainevent/_csr/delay_test.py`)

Dense numpy reference loop. Collisions, wraparound, homo/hetero, bool / int /
float events with negatives, binary and weighted modes, traced `t` in a
`lax.scan` mini network (deliver, read slot `t mod L`, clear it), invalid
delays, scalar delays, every available backend, units, `CSR.delay_deliver`,
`jax.grad` / `jax.jvp` / dot-product identity versus a dense JAX reference,
`vmap`, float64 under x64, GPU-only tests (`requires_gpu`): f16/bf16, int64
indptr, larger random graph versus `jax_raw`.

## Benchmark

`dev/benchmarks/csr_delay_deliver_bench.py`: Brunel-like network, 12 500
neurons, indegree 1250 (15.6 M edges), L = 200, delays uniform in `[1, 199]`,
firing probability 1-5 % per step, float32 hetero weights. Per-step time of
(a) `csr_delay_deliver` (cuda_raw / numba), (b) the `jax_raw` scatter,
(c) the scalar-delay `BinaryArray(spikes) @ CSR` baseline, plus
`jax_event`, an event-driven pure-JAX version in the style of brainpy.state
(compact active rows with a fixed capacity, gather their edges, one
scatter-add; no overflow `cond`), and `none` (scan + slot read/clear only).

Each step runs inside one jitted `lax.scan`: deliver, read slot `t % L`,
clear it. Reading a slot and clearing it in the same step makes XLA copy
the whole ring twice per step (it cannot otherwise order the read before the
in-place clear); `lax.optimization_barrier((out, ring))` between the read
and the clear removes both copies. The harness uses it, and users writing
such loops should too.

Results (per-step time, best of several runs; RTX 3090; CPU numbers on one
pinned core, `JAX_PLATFORMS=cpu taskset -c 0`; jax 0.12.0, float32 hetero
weights, 15 625 000 edges):

GPU

| rate | csr_delay_deliver cuda_raw | jax_raw scatter | jax_event | scalar `BinaryArray @ CSR` | none |
|---|---|---|---|---|---|
| 1% | 0.040 ms | 2.40 ms | 0.197 ms | 0.024 ms | 0.008 ms |
| 2% | 0.060 ms | 2.40 ms | 0.298 ms | 0.029 ms | 0.007 ms |
| 5% | 0.131 ms | 2.40 ms | 0.613 ms | 0.041 ms | 0.007 ms |

CPU (one core)

| rate | csr_delay_deliver numba | jax_raw scatter | jax_event | scalar `BinaryArray @ CSR` | none |
|---|---|---|---|---|---|
| 1% | 0.727 ms | 63.9 ms | 0.864 ms | 0.954 ms | 0.008 ms |
| 2% | 1.326 ms | 63.6 ms | 1.426 ms | 1.079 ms | 0.008 ms |
| 5% | 2.903 ms | 65.1 ms | 3.865 ms | 1.552 ms | 0.008 ms |

Reading:

- GPU: the fused kernel is 4.7-5x faster than the event-driven JAX
  scatter and 18-60x faster than `jax_raw`, but still 1.7-3.2x the scalar
  baseline. The remaining gap is the atomics themselves: the scalar path
  adds into a 50 KB vector that stays in L2, the delayed path into a 10 MB
  ring that does not fit the 6 MB L2 of the 3090.
- CPU: `numba` is 1.1-1.3x faster than `jax_event`; relative to the scalar
  baseline it is 0.76x at 1 %, 1.2x at 2 % and 1.9x at 5 % (random writes
  into a 10 MB ring, as the issue noted). `jax_raw` visits every edge every step
  and is only a reference / AD backend.
- The checksum of all arrivals agrees across `cuda_raw`, `numba`,
  `jax_raw` and `jax_event` (printed by the script).
