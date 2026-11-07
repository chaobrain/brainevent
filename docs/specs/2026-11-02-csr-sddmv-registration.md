# CSR SDDMV registration

## Scope

Register a CSR sampled weight-gradient operator following the TCSR SDDMM
packaging pattern. Do not replace the existing ordinary/indexed CSR transpose
rules in this change. The operator computes other operators' first derivatives;
it intentionally has no derivative rule of its own.

## Decisions

- `csr_sddmv_dweight(events, ct, indices, indptr, transpose=..., weighted=...)`
  returns one value per physical CSR slot, in `ct.dtype`. Convenience binary and
  weighted wrappers share the same registered primitive.
- The default GPU backend dispatches `transpose=True` to hybrid CUDA and
  `transpose=False` to cuSPARSE SDDMM with contraction size one. Register
  explicit `cuda_raw`, `cusparse`, and a portable `jax_raw` reference backend.
- Preserve event selection: inactive events contribute exactly zero, including
  with NaN/Inf cotangents. Put cotangent checks and exact selection fallback
  inside SDDMV. Determine activity before converting event values to `ct.dtype`.
- Cover float16, bfloat16, float32, float64 cotangents and boolean, integer,
  and floating events. Add native half/bfloat16 weighted entry points. For
  cuSPARSE, compute half/bfloat16 products in float32 then cast back; this also
  avoids relying on BF16 support in CSR SDDMM. Cast event amplitudes to the
  cotangent dtype before promotion, preserving the existing CSR arithmetic.
- Allocate zero output and task scratch per invocation. Use the same captured
  HybridConfig for static capacity and compilation; do not read traced row
  offsets on the host. Package all required headers locally.
- Keep the existing binary shared-weight reduction unchanged. It uses a
  unit-weight forward product followed by an inner product for finite `ct`,
  and exact per-slot selection followed by summation otherwise. Future
  integration must retain canonical-order scatter for indexed weights.

Audit finding: finite `ct` alone does not make the shared-weight fast reduction
safe from intermediate overflow. Four f32 events of `1e38` accumulating into
one output make the unit-weight product infinite; with `ct=0` the existing MV
gradient is NaN, while per-slot multiplication followed by summation gives zero.
Record this for future integration; changing the current transpose rules is
outside the user-confirmed scope of this change.

## Verification

Check registered backend selection in lowered IR and execute both GPU paths.
Compare against per-slot selection across value/event types, both orientations,
NaN/Inf, conversion underflow, empty/duplicate structure, heavy rows, traced
structure, vmap, and shard_map. Confirm explicit differentiation is unsupported.
Check type annotations and wheel contents. Existing binary gradient tests remain
unchanged because their call sites are outside this change's scope.
