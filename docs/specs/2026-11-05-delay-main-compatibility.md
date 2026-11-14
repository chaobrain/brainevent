# Delay input compatibility with main

Date: 2026-11-05. Compatibility baseline: `main@c099629`.

## Goal

Restore the existing delay input types, units and differentiation behavior at
the prepared `delaycsr` boundary. Keep the region-delay schedule, CUDA ABI and
kernels unchanged. This work does not change delay timing or the strict
`with_data` replacement contract.

## Decisions

- Preserve canonical weights and their unit. Construction and replacement own
  copies of caller-owned NumPy storage; preparing a delivery must not replace
  `self.data` with a converted execution array.
- The ring mantissa dtype determines execution storage: float16, bfloat16,
  float32 or float64. Convert temporary weights to that dtype and prepare the
  weighted task-value queue in the same dtype. Explicit rings need not have
  the canonical weight dtype.
- Split weight and event units, convert the ring to their product unit, and
  restore that unit on the result. Incompatible ring units are errors. An
  external-ring delivery leaves the layout's owned ring, head, workspace and
  canonical data unchanged.
- Keep the original event mantissa until primitive binding, including inside
  JVP rules. JAX selects using the original `event != 0` and casts only the
  weighted amplitude to the ring dtype. CUDA adapts binary event types and
  casts weighted amplitudes only inside lowering.
- Weight VJP selects using the original event, multiplies in the cotangent
  dtype and returns the weight dtype. Event VJP retains the existing weighted
  derivative / binary straight-through rule and returns the event dtype.
- Static array validation belongs at primitive abstract evaluation so raw
  binds, class calls, JVPs and the public prepared wrapper share those checks.
  The public wrapper also validates boolean flags before binding, since JAX
  parameter caching can equate `0`/`1` with `False`/`True`. Keep object-specific
  state checks, native FFI checks and dynamic head/task checks. Validate delay
  membership before changing sorted structure; commit
  a cleared ring and replacement workspace only after successful projection
  construction. The existing explicit-time `forward` head assignment remains.
- `with_data` still requires an initialized object, a weight array in its
  current edge order, and exactly matching shape, dtype and unit. It does not
  switch between shared and per-edge weight representations.

## Existing backend difference

The baseline already differs for weighted events that underflow when cast to
the ring dtype. With float16 storage, a float32 event `1e-10` and an infinite
weight, JAX delivers NaN while CUDA delivers zero. An event JVP with the same
tiny tangent has the same difference. A weight VJP with an infinite cotangent
uses original event activity and returns NaN on both backends. Preserve these
baseline behaviors; this change does not promise cross-backend equality for
this case. Unifying them would require a separate CUDA activity operand or
native support for the original event dtype, including long-row extraction.

## Implementation and verification

1. Centralize prepared-primitive validation and move event conversion to the
   appropriate lowering; remove the early JVP event cast.
2. Normalize units and temporary execution weights at the object boundary,
   with ring-typed workspace and preserved canonical data/state ownership.
3. Compare against the baseline for binary/weighted mixed event types,
   homogeneous/per-edge weights, unit conversion, JIT, JVP and VJP. Include
   zero-event NaN/Inf isolation and the underflow cases above on CPU and CUDA.
   Check raw primitive validation, short/long rows, caller-array mutation,
   external-ring state isolation and unchanged `with_data` rejection cases.

## Validation results

- JAX 0.11.0, full Delay suite on CUDA with slow tests enabled: 820 passed,
  2 skipped (CPU-default routing and the opt-in 20–30 GB allocation case).
- CPU suite: 453 passed, 368 skipped. After the final Python-list ring
  precision correction, constructor tests: 89 passed, 3 skipped, 1 deselected.
- Two-device CPU sharding checks: 2 passed. The three changed production
  Python files have 94% combined line coverage on the full CUDA suite.
- `mypy --no-site-packages brainevent/`: 178 source files passed.
- `git diff --check` passed.
