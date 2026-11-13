# TCSR inherited capabilities and API compatibility

## Goal

Verify the new TCSR against CSR's public capabilities and reconcile old
argument names. No new pp_prop/BPSA implementation or gradient semantics.

## Decisions

- Keep inherited CSR method names, signatures and limitations. Test actual
  dt2t, plasticity, solve, conversions and delay behavior against CSR.
- Use `csr`, `csc`, `mat` and `backend` as canonical construction arguments.
  Accept old `source` and `binary_backend` keywords through thin adapters.
  Map legacy `jax` to `jax_raw`; reject conflicting backend choices or two
  source arguments. Preserve source configuration and canonical edge order.
- Restore `fromcsr`/`fromcsc` as wrappers around sorted construction and
  `materialize_tcsc_mirror`/`has_tcsc_mirror` as public mirror aliases.
  No additional storage, transpose class or inherited-method overrides.
- Document sorted external edge state, `.T` type behavior and the existing
  parent-owned backward option without adding new gradient behavior.

## Steps and verification

1. Add failing compatibility tests, then implement argument/factory adapters.
2. Exercise inherited APIs with keyword calls, units, homogeneous weights,
   canonical edge order and suitable JAX transformations.
3. Update the API reference, current README and changelog.
4. Run focused CPU and CUDA regressions, JAX 0.9.0 compatibility, new-code
   mypy and Python coverage. Retain parent errors for unsupported inputs.

## Results

- All six documented overrides remain implemented. No additional inherited
  operations needed overrides: 20 general inheritance tests and 13 delay
  tests pass on both CPU and CUDA, and on JAX 0.9.0 CPU.
- Compatibility tests first reproduced missing aliases/factories. All 21
  now pass on CUDA, including canonical keywords, conflicting arguments,
  units and CUDA parent MM after overriding a JAX-prepared backend.
- Checked construction prepares GPU configuration using concrete weights'
  actual placement when host metadata carries no device information. Trusted
  assembly remains host-free; cross-backend configuration rebuilding is
  explicit. No matrix arrays are implicitly moved by rebuilding configuration.
- Combined regression: 186 passed on JAX 0.11.0; JAX 0.9.0 CPU: 145 passed,
  41 GPU cases skipped. New-code mypy and patch checks pass.
- COO conversion preserves every edge. The current downstream CUDA COO
  densification does not reliably sum duplicate coordinates; the CSR/TCSR
  shared limitation and alternative CSR densification are documented.
- API reference examples pass doctest. Full Sphinx build was not run because
  Sphinx/docutils are absent from the development environment.
