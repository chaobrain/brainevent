# CSR pp-prop weight gradient

## Goal

Give CSR event products the TCSR `backward_algorithm="pp_prop"` rule: a
`BinaryArray` still multiplies by its activity mask in the forward pass, but
its weight cotangent uses the numeric event values. The default `"bptt"`
continues to use the activity mask.

## Decisions

- Store the algorithm on the shared compressed sparse base so CSR's CSC
  transpose and converted views preserve it. Carry it in pytree auxiliary
  metadata and all structure-preserving constructors.
- Add the same optional argument to direct and indexed CSR MV/MM services and
  primitives. The `weighted` flag remains the forward event mode; only the
  SDDMV/SDDMM weight cotangent selects `weighted` when `pp_prop` is requested.
- `WeightedArray` always uses value-weighted forward and weight cotangents,
  independently of `backward_algorithm`. Event cotangents and weight JVPs
  retain their existing rules. Thus `pp_prop` on a binary forward is an
  intentional surrogate weight VJP for non-binary numeric events.

## Verification

- Compare binary forward outputs and weight cotangents for `bptt`/`pp_prop`
  across MV/MM, both multiplication directions, and homogeneous/heterogeneous
  weights.
- Exercise indexed mirror routing, pytree round-trip, conversion/transpose,
  invalid algorithm rejection, and the `WeightedArray` override.
