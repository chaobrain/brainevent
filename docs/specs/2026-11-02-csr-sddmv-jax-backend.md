# CSR SDDMV JAX backend

This replaces the nontransposed backend decision in
`2026-11-02-csr-sddmv-registration.md` following the user's request to use JAX
instead of a custom cuSPARSE CUDA wrapper. Existing CSR gradient consumers
remain unchanged.

- Default GPU dispatch: `transpose=True` uses the existing hybrid CUDA
  kernel; `transpose=False` uses the registered `jax_raw` implementation.
- Reuse `brainevent._sddmm.sddmm_coo_indices` with inner dimension one.
  In installed JAX 0.11.0, `bcoo_dot_general_sampled` lowers to JAX gathers
  and multiplication/contraction, not `cusparseSDDMM`. Name the backend
  `jax_raw` accordingly and remove the direct cuSPARSE C++/FFI files and
  library discovery code. JAX's cuSPARSE-backed SpMV/SpMM are distinct ops.
- Keep the finite-`ct` check and exact inactive-event selection in SDDMV.
  Preserve all existing value/event types, duplicate physical slots, empty
  structure, and mixed-precision activity. SDDMV itself still has no AD rule.
- Reuse the existing sampled helper's shard_map handling rather than adding
  a second implementation. Verify lowering and execution with the CUDA
  extension loader disabled for the nontransposed path; run numerical,
  dtype, vmap and shard_map regression tests on GPU and CPU.

The earlier implementation treated a backend registration request as a reason
to add another native binding. Check existing JAX/project primitives and their
actual lowerings before introducing library bindings in future changes.
