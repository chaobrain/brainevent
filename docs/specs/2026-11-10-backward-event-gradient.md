# Explicit event weight-gradient modes

## Goal

Replace `backward_algorithm='bptt'/'pp_prop'` with `'binary'/'weighted_float'`
throughout CSR, CSC and TCSR. Default to `'weighted_float'` so event weight
gradients use numeric event amplitudes when entering SDDMM.

## Decisions

- `binary` selects activity-mask weight gradients. `weighted_float` selects
  the existing SDDMM `mode='weighted'`, retaining inactive-event suppression.
  Ordinary floating-point CSR gradients keep `mode='float'`.
- Value-weighted forward events always retain amplitude-based gradients.
  Binary forward products keep their activity-mask semantics.
- Reject the old names rather than retaining aliases. Update constructor,
  factory, primitive-call and transformation fallback defaults together.
- Preserve the selected mode through matrix conversions, mirrors, PyTrees,
  TCSR delegation and shared-weight reductions.

## Implementation and verification

1. Reproduce the old default's mask gradient using non-unit numeric events.
2. Update shared validation, defaults, transforms and API documentation.
3. Test defaults and explicit modes for MV/MM, both directions, indexed and
   direct layouts, shared weights, invalid names and matrix lifecycle.
4. Run affected CSR/TCSR and SDDMM tests, available GPU regressions and the
   repository type/style checks; record any environment limitations.
