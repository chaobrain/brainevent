# A conceptually-universal common-API contract on DataRepresentation

`DataRepresentation` now declares one contract — the conversion methods
(`todense`, `fromdense`, `tocoo`, `tocsr`, `tocsc`) and the neural protocol
(`yw_to_w`, `yw_to_w_transposed`, `update_on_pre`, `update_on_post`) — covering
*every* operation meaningful for a generic sparse weight matrix, even where a
particular family cannot support it. Each contract method is a documented
`NotImplementedError` stub on the base; a subclass that cannot meaningfully
implement one **deliberately refuses** by raising the new
`UnsupportedOperationError(BrainEventError)`. A parametrized coverage test
asserts that every concrete subclass either overrides each contract method or
refuses it, so no subclass silently inherits a bare stub.

## Status

accepted

## Considered Options

- **`abc.abstractmethod` contract** — rejected: forces all 10 concrete classes
  to redeclare every method, departs from the codebase's plain-class +
  `NotImplementedError` convention (inherited from `saiunit.sparse.SparseMatrix`),
  and complicates the `classmethod`/pytree interplay.
- **Plain `NotImplementedError` stubs only** — rejected: a silently-inherited
  stub is indistinguishable from a deliberate refusal, so the "is this operation
  suitable for this family?" verdict would live only in prose, not in code.
- **Chosen: stubs + `UnsupportedOperationError` + coverage test** — keeps the
  existing convention, encodes the suitability verdict at runtime (greppable,
  catchable), and enforces capability coverage via one test rather than the type
  system.

## Consequences

- "Not yet implemented" (`NotImplementedError`) is now semantically distinct
  from "structurally meaningless for this representation"
  (`UnsupportedOperationError`). The JIT-connectivity family refuses
  `fromdense` (cannot recover `(prob, seed)`), `yw_to_w`/`yw_to_w_transposed`
  (weights are distribution parameters, not per-synapse), and
  `update_on_pre`/`update_on_post` (no per-synapse plastic weight) — each message
  points at `.tocsr()` for an explicit materialized path.
- Conversion method spelling is canonicalised to the no-underscore scipy/saiunit
  form (`todense`/`fromdense`/`tocoo`/`tocsr`/`tocsc`). The fixed-num-connection
  family's `to_csr`/`to_csc`/`to_dense` become deprecated aliases.
- `diag_add` and `solve` stay off the contract — they require a square,
  invertible matrix and are specific to the compressed-sparse family.
