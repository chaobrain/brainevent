# Precomputed delay-major layout for `csr_delay_deliver` (#203, #205)

Branch: `feat/csr-delay-deliver`.

## Problem

brainpy.state's target for per-edge-delay delivery is at most 1.5x the cost
of a scalar-delay step (`BinaryArray(spikes) @ CSR`). On the GPU the
row-major kernel (`csr_delay_deliver`, cuda_raw) is 1.7-3.4x: each active
edge adds into a random slot of an `(L, n_post)` ring that does not fit in
L2 (10 MB for the Brunel benchmark, 6 MB L2 on an RTX 3090), and random
sector traffic caps the kernel at about 11 G sectors/s. Runtime
reorganisations (column passes, shared-memory tiles, two-phase binning, L2
cache hints) reach at best ~1.75x; see the measurements below.

## Design

Sort the edges of every row by delay once, and record where each delay's
segment starts. A step then processes all active rows delay by delay:
every write for delay `d` lands in ring row `(t + d) mod L` (50 KB in the
benchmark), which stays in L2 while it is updated and is written back
whole. Delays become implicit, so a step reads 8 bytes per edge (index and
weight) instead of 12.

```python
layout = brainevent.CSRDelayLayout.build(data, indices, indptr, delays, n_slots=L, shape=(n_pre, n_post))
layout = csr.delay_layout(delays, n_slots=L)            # same, from a CSR
ring = brainevent.csr_delay_deliver_layout(ring, events, layout, t)
ring = layout.deliver(ring, events, t)                    # method form
```

`CSRDelayLayout` (pytree; every array is a leaf, `shape` and `n_slots` are
static):

| field | shape / dtype | content |
|---|---|---|
| `data` | `(1,)` or `(nse,)` | weights in layout order (homogeneous weights unchanged) |
| `indices` | `(nse,)` int32 | column indices in layout order |
| `indptr` | `(n_pre + 1,)` | unchanged |
| `delays` | `(nse,)` int32 | delays in layout order, validated to `[1, L)` |
| `delay_ptr` | `(n_pre, L)`, indptr dtype | `delay_ptr[i, d]` = first position of delay `d` in row `i` |
| `perm` | `(nse,)` int | layout position -> original edge |

Edges are ordered by `(row, delay)`; ties keep the original order, so a
row's column order is preserved within each delay. `build` runs on the host
with NumPy and needs concrete `indices`, `indptr` and `delays`; `data` may
be traced. Memory: `delay_ptr` costs `n_pre * L * itemsize` (10 MB for the
benchmark, about 5 % of its CSR).

Weights live in layout order, so `jax.grad` and plasticity act on
`layout.data` directly. `layout.with_data(w)` takes weights in the original
edge order (`w[perm]`) and `layout.original_data()` returns them in the
original order.

## Primitive

`csr_delay_deliver_layout_p`, operands `(ring, events, data, indices,
indptr, delays, delay_ptr, t)`, static `shape`, `weighted`; outputs the ring
(aliased) and, on cuda_raw, a scratch buffer.

- `cuda_raw` (GPU default), `csr_delay_deliver_layout.cu`: compact the
  active rows into the scratch buffer (`n_pre + 1` int32, the last entry is
  the count), then a persistent grid takes items `(d, active row)` in
  delay-major order, 16 lanes per item, and adds into `ring[(t + d) mod L]`
  with atomics. Weight dtypes f32/f64/f16/bf16, homogeneous/heterogeneous,
  bool/float/weighted events; int32/int64 indptr.
- `numba`, `jax_raw`: the existing `csr_delay_deliver` kernels on the
  layout arrays (`delay_ptr` is unused). Results are identical to
  `csr_delay_deliver` on the same layout arrays.
- JVP, transpose and vmap: the same formulas as `csr_delay_deliver`
  (the operation is identical on the reordered arrays); tangents go through
  the layout primitive, cotangents through the per-edge formula.
  `indices`, `indptr`, `delays`, `delay_ptr` and `t` are not
  differentiable.

Activity is `x != 0`; inactive rows contribute nothing (NaN/Inf weights in
inactive rows are isolated); NaN events are active.

## Tests (`delay_layout_test.py`)

- `build`: ordering, `delay_ptr` against a brute-force count, `perm`
  round trip (`with_data` / `original_data`), homogeneous data, scalar
  delays, traced data, validation errors (out-of-range or traced delays,
  shape mismatches), int64 indptr.
- Forward: equal to `csr_delay_deliver` on the original CSR for every
  backend, binary (bool/int/float events) and weighted events, homo/hetero,
  f32/f64/f16/bf16, empty rows, rows with one delay, `t` negative and
  beyond int32 range, collisions on the same `(slot, post)`.
- NaN isolation: NaN/Inf weights in inactive rows; active-row NaN
  propagates.
- AD: `jax.grad` and `jax.jvp` w.r.t. ring, data, events vs the original
  primitive; vmap over events and ring.
- `CSR.delay_layout`, units.

## Benchmark

`examples/csr_delay_deliver_benchmark.py` gains `layout/<backend>`
variants. Target: `layout/cuda_raw` <= 1.5x `scalar` at 1 %, 2 % and 5 %
firing on the RTX 3090; numba (`deliver/numba`) <= 1.5x on one CPU core.

## Measured prototypes (RTX 3090, kernel time per step, 12 500 x 12 500, K = 1 250, L = 200)

| design | 1 % | 2 % | 5 % |
|---|---|---|---|
| budget for <= 1.5x | ~28 us | ~35 us | ~57 us |
| row-major kernel (current) | 30 | 52 | 122 |
| column passes (sorted rows) | 27 | 41 | 69 |
| two-phase bin + shared-memory apply | 29 | 39 | 66 + gaps |
| L2 cache hints | 28 | 47 | 112 |
| delay-major layout, 16 lanes per item | 20 | 27 | 45 |

## Measured (end to end, `examples/csr_delay_deliver_benchmark.py`)

| rate | deliver cuda_raw | layout cuda_raw | scalar | layout / scalar |
|---|---|---|---|---|
| 1 % | 0.042 ms | 0.031 ms | 0.030 ms | 1.03x |
| 2 % | 0.065 ms | 0.038 ms | 0.032 ms | 1.19x |
| 5 % | 0.136 ms | 0.060 ms | 0.047 ms | 1.28x |

CPU (one core): `layout/numba` takes the same time as `deliver/numba`
(0.379 / 0.722 / 1.726 ms vs scalar 0.792 / 0.950 / 1.411 ms).
