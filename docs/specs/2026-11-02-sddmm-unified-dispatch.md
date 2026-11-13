# Unified SDDMM dispatch

## Contract

`brainevent._sddmm.sddmm(A, B, indices, indptr=None, *, mode='float',
transpose=True, backend=None)` returns values in physical sparse-slot order.
Vectors represent a rank-one outer product; matrices use `(rows, batch)` and
`(batch, cols)`. Dispatch depends on the static contraction size, not an outer
vmap axis: one goes to `_mv`, all other sizes (including zero) to `_mm`.

Without `indptr`, indices are `(nse, 2)` COO coordinates. With `indptr`, columns
are int32 CSR indices and offsets int32/int64. No sorting or coalescing occurs.
`mode='float'` preserves ordinary multiplication and the legacy JAX derivative
behavior. Event modes `binary` and `weighted` identify events on the left when
`transpose=True`, on the right otherwise, preserving activity before casting.

## Layers

- The package facade exports the dispatcher and compatibility wrappers.
  `sddmm_indices`, `sddmm_coo_indices` and `sddmm_bcoo` preserve BCOO returns.
  CSR event wrappers return arrays and all enter the same dispatcher.
- The dispatcher validates/normalizes inputs and chooses MV or MM. It never
  chooses a backend. Leaf modules never import/call the dispatcher.
- MV owns the rank-one JAX implementation and existing hybrid CUDA event
  primitive. Native CUDA is available for transposed CSR event products.
- MM is callable now, with a complete JAX sampled-product implementation and
  a `_cuda_kernel` placeholder. CUDA is not registered for MM; explicit CUDA
  requests fail clearly. Automatic MM selection uses JAX.
- Both leaf JAX implementations preserve shard_map semantics. Event paths
  own finite-cotangent checks and exact per-slot selection; ordinary products
  remain unmasked. Existing event-gradient helpers need no derivative rules.

## Migration and scope

Remove the old `_sddmm.py` after moving its implementation into the leaves.
Repair relocated MV configuration imports, CSR exports, tests and package
data. Keep the existing CSR AD consumers and TCSR-specific CUDA/tile/permutation
interfaces unchanged: legacy generic SDDMM callers enter the new dispatcher
through their existing wrappers. TCSR-specific integration remains a subsequent
adapter migration as agreed in the implementation plan.

## Verification

Run legacy generic SDDMM tests, relocated MV tests, and new dispatch/MM tests.
Cover batch 0/1/n, vectors and singleton matrices, ordinary/binary/weighted
semantics, both directions, nonfinite ct, mixed types, empty/duplicate/OOR COO
slots, jit/vmap/shard_map, and the legacy JAX derivatives. Verify actual MV CUDA
lowering, JAX-only MM execution with CUDA loading disabled, and explicit MM CUDA
errors. Check mypy and a fresh wheel outside the source tree.
