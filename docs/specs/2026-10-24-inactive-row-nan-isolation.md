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

cuSPARSE and cuBLAS take the weights as a library operand. Masking the
operand per call (`where(active[row(e)], data, 0)`) only works for mv: in mm
the activity differs per batch column, so no single masked operand is right
for every column, and for TCSR mv it cost 3.5x (0.108 -> 0.385 ms, mostly the
row-id expansion). The library paths therefore keep their fast path when it
is exact and fall back otherwise: if every weight is finite, `0 * w` is an
exact `0` and the library op runs as before; if not, `lax.cond` switches to
the select-based `jax_raw` kernel. Homogeneous binary products select on the
event count (`where(count != 0, count * w, 0)`). Measured (RTX 3090, TCSR,
20000^2, 1 % density, 5 % active): mv hetero 0.108 -> 0.145 ms (finiteness
check), mm batch 16 0.455 -> 0.47-0.52 ms; event-driven cuda_raw and numba
paths unchanged.

## Helpers

`brainevent._op.util.select_active(active, values)` (`where(active, values,
0)` keeping the dtype) and `active_sum(r, events, *, weighted=False)` (scalar
weight-parameter cotangents summed over active events only). `event_factor`
stays for forward factors whose other operand is known finite.

## Tests

Per format, per backend (CPU: `jax_raw`, `numba`; GPU: `cuda_raw`,
`cusparse`, `cublas`, TCSR paths), mv and mm, both `transpose`, homo and
hetero where applicable, binary (bool and float events) and weighted events:
put `NaN`, `+Inf` and `-Inf` weights only on entries whose event is inactive
and assert the output is finite and equals the dense reference computed with
those weights replaced by `0`. Plasticity: inactive edges unchanged when the
trace holds NaN. One regression test that the active-row NaN still propagates.
