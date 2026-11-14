# SDDMM CUDA assembly

## Scope

Connect the existing MV/MM binary and weighted CUDA sources to the unified
SDDMM dispatcher. Keep native algorithms, public signatures and CSR reverse
mode consumers unchanged. GPU auto uses CUDA for transposed CSR events and
JAX for nontransposed events; explicit JAX supports both. Ordinary/COO products
keep JAX. Empty outputs and batch zero need no native call. MM auto falls back
to JAX beyond the native batch/column limits; explicit CUDA rejects them.

## Assembly decisions

- MV loads the renamed sources with local/shared header include paths. Weighted
  calls supply the additional task-value operand/result and its buffer alias.
- Both leaves share the native dtype suffix table, retain original binary
  activity before casting, and preserve weighted mixed-float activity with
  nonfinite cotangents through exact JAX selection.
- MM transposes logical `(units, batch)` operands to the native batch-first
  layout. It aliases a zeroed dW input to output 0 and allocates private uint32
  workspace and int32 selected-row outputs. Half types add float32 accumulators.
- A host `make_plan(indptr)` cannot consume traced CSR pointers. The adapter
  instead uses fixed group 32 and chunk min(batch, 64), computes the packed
  `length > 512` row mask in JAX, and bounds tasks from shape metadata. With
  `h = min(rows, nnz // 513)`, per-chunk capacity is
  `chunk * floor((nnz + 511*h) / 512)` when long rows are possible.
  The native launcher only tests short/long counts for zero, so adapter counts
  are presence flags enabling every possible schedule. For one row the shape
  determines which schedule is needed. Actual device counters/mask drive work.
  This trades topology-based tuning for dynamic CSR support without host copies.
- Include all new `.cuh` files in wheels and compilation cache dependencies.

## Verification

First reproduce the stale MV loader and missing MM registration in GPU tests.
Then test explicit/automatic CUDA lowering, both JAX directions, all value
dtypes, mixed events, NaN/Inf, empty/duplicate/heavy rows, zero batch, single
rows, mask-word and chunk boundaries, changing CSR under one compilation,
repeated inactive calls, vmap and shard_map. Run the SDDMM suite on CPU/GPU,
mypy, and a wheel import/CUDA smoke test outside the source tree. Native atomics
permit reduction-order differences; no performance claim is made.
