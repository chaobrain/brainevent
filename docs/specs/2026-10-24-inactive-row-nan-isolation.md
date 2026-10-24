# Inactive events contribute exactly zero (NaN/Inf weight isolation)

Branch: `fix/inactive-row-nan` (stacked on `chore/test-hygiene`).

## Problem

An inactive event (`x == 0`, or `False`) must not influence the output. Today
whether a NaN/Inf weight in an inactive row (or column, for the other
transpose direction) reaches the output depends on the backend:

- event-driven kernels (numba, CUDA scatter/hybrid, dense CUDA) skip inactive
  entries, so the output is finite;
- gather-style CUDA kernels, `jax_raw`, cuSPARSE and cuBLAS compute
  `w * 0`, which is `NaN` for `w = NaN` or `w = ±Inf`.

The same model therefore gives finite results on one backend and NaN on
another (`2026-10-24-weighted-array.md`, "Activity in weighted kernels").

## Contract

For every event operation (binary and weighted events, every format, every
backend, mv and mm, both `transpose` directions):

- **Forward.** Inactive events contribute exactly `0`, whatever the weight
  values they would multiply (finite, NaN, ±Inf). Implementation rule: select,
  never multiply by a 0/1 mask — `where(active, w, 0)` (or `where(active, w*x,
  0)` for weighted events), or skip the entry.
- **Plasticity.** Edges whose driving spike is inactive are returned
  unchanged (bit-identical), whatever the trace value (`w + where(m, trace, 0)`,
  not `w + m * trace`).
- **Delay delivery.** Same as forward: inactive rows add nothing to the ring.
- **Gradients.** The weight cotangent of an inactive row is exactly `0`
  (select, not `ct * 0`). Event cotangents/JVPs are unchanged (straight-through
  `W^T ct` legitimately sees every weight).
- **Not covered.** Active events times a NaN/Inf weight stay NaN/Inf; dense
  float (non-event) products keep IEEE semantics; JITC weights are generated
  from finite parameters, so only a NaN/Inf *parameter* (`w`, `w_loc`, ...)
  applies, and the same rule holds for it.

## Backends without a select hook

cuSPARSE and cuBLAS take the weights as a library operand. They mask the
operand first: CSR `data` with `where(active[row(e)], data, 0)` (O(nse)),
dense `W` with `where(active[:, None], W, 0)` (O(n·m), the same order as the
product). Measure and record the cost.

## Tests

Per format, per backend (CPU: `jax_raw`, `numba`; GPU: `cuda_raw`,
`cusparse`, `cublas`, TCSR paths), mv and mm, both `transpose`, homo and
hetero where applicable, binary (bool and float events) and weighted events:
put `NaN`, `+Inf` and `-Inf` weights only on entries whose event is inactive
and assert the output is finite and equals the dense reference computed with
those weights replaced by `0`. Plasticity: inactive edges unchanged when the
trace holds NaN. One regression test that the active-row NaN still propagates.
