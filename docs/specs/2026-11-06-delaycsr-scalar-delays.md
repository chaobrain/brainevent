# Scalar delays in delaycsr

## Goal

Let `delaycsr` and `CSR.delay_layout` accept one integer delay shared by all
edges, while retaining per-edge delay arrays and the existing prepared layout.

## Decisions

- Keep `edge_delays` as the only delay input to `delaycsr`. A concrete integer
  scalar or one-element array broadcasts to `nnz` edges, matching the `main`
  branch's `CSRDelayLayout.build` behavior. An array of shape `(nnz,)` keeps
  its per-edge meaning. No delay wrapper type is needed.
- Normalize and validate delays once in `delaycsr._validate_edge_delays`, which
  is also used by `sort_by_delay`. Sorting, metadata, and delivery continue to
  use the resulting per-edge array.
- Keep the existing `[0, buffer_slots)` range. For an empty graph, broadcast
  produces an empty array and the existing default `delay_steps=(0,)` remains.
  Explicit `buffer_slots` still determines ring length.

## Steps

1. Update constructor and `CSR.delay_layout` documentation.
2. Broadcast a valid scalar or one-element array to a writable per-edge array.
3. Test construction, delivery equivalence, empty graphs, and invalid values.

## Verification

Run the delaycsr initialization and delivery tests with the CPU `jax_raw`
backend, including the new scalar cases.
