# Direct construction of prepared delay CSR

## Goal

Construct a complete `delaycsr` from a caller-sorted CSR and supplied delay
metadata, without sorting, counting edges or preparing a hybrid schedule.

## Decisions

- Add `delaycsr.from_prepared` while keeping the ordinary constructor intact.
  Accept a CSR, a three-array tuple, or separate weights, indices and indptr.
- Require static `delay_steps` and `capacity`, plus `delay_count`, `delay_off`
  and `row_task_count`. Do not derive these arrays or copy them to the host
  during default static validation. Preserve independent index/indptr widths.
- Each row stores consecutive delay segments in delay-step order, with sorted
  columns within each segment and aligned weights. No per-edge delays remain.
- Keep the current hybrid schedule: degree <=128 uses zero tasks; otherwise
  sum per-segment ceilings in chunks of 4096. Capacity covers all rows firing.
- Default ring length to the maximum delay plus one. Accept optional ring,
  head and binary/weighted workspace; otherwise allocate fresh buffers.
- Reuse JAX operands and snapshot NumPy operands. Retain ring precision/unit
  behavior, backend inheritance, pytree state and primitive safety checks.
- `validate='static'` checks the shape/dtype contract and static parameters;
  the existing head-range check is retained. `validate='full'` reads concrete
  layout arrays to check consistency before allocating default buffers. It
  checks declared segments but cannot recover or verify original delay labels.

## Steps

1. Assemble base CSR bookkeeping from trusted parts, bypassing both constructors.
2. Attach supplied layout arrays and execution state after validation.
3. Document the contract and test state, ownership and transformation behavior.

## Verification

- Forbid constructor, sorting and metadata-generation calls in a focused test.
- Compare delivery with ordinary initialization for binary/weighted modes,
  empty graphs, segment chunk boundaries and overallocated workspace.
- Exercise static/full errors, NumPy isolation and device operand reuse.
- Compare JIT scan outputs, JVPs and reverse gradients with ordinary objects.
- Run the delay regression suite, scoped coverage, doctests and mypy.