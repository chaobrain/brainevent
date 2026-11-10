# Automatic TCSR tiles and public matmul overrides

## Goal

Implement the latest `_tcsr/dev/README.md`: construct `TCSR(csr)` once, prepare
all primary geometry, lazily prepare mirror geometry after CSR supplies its
mirror, and preserve CSR call and input semantics through public matmul overrides.
No new pp_prop/BPSA algorithms.

## Decisions

- Store unique geometry by direction, actual width and offset dtype. Enumerate
  planner batches 0..TARGET_BLOCKS and both accumulator sizes to cover all
  supported future batches. Keep device tile arrays as dynamic pytree leaves.
- Initialize primary outside JIT. A thin `_mirror()` override calls CSR first,
  then atomically publishes complete mirror geometry for concrete structure.
  Traced missing geometry follows CSR; trusted reconstruction never prepares.
- Override both public matmul methods and remove the old event-product override.
  Retain the original operand for every parent fallback, including lowering
  choices for platform, backend and device resource limits.
- Preserve the existing compatibility factories and argument aliases. Existing
  explicit preparation remains optional; ordinary calls never depend on it.
- Concrete structural results prepare new primary geometry; traced results
  discard obsolete geometry and retain the applicable parent behavior.

## Steps and verification

1. Add automatic/lazy lifecycle regressions and replace batch-keyed cache usage.
2. Implement full geometry preparation and public-operator dispatch.
3. Implement actual-target resource selection before native execution.
4. Verify CPU/CUDA numerical behavior, dispatch, units, gradients, JIT/scan/vmap,
   placeholders, inherited methods, invalidation and JAX 0.9.0 CPU compatibility.
5. Run focused mypy, meaningful >90% coverage, native memory checks where
   applicable, and update current docs to the implemented contract.

## Implementation results

- `TCSR(csr)` now initializes complete primary geometry automatically.
  `_mirror()` calls CSR first, then publishes complete concrete mirror
  geometry lazily. Missing traced geometry follows the parent route.
- Public left/right matmul overrides retain the original operand for every
  fallback. The former event-product override is removed. Concrete sliced
  and diagonal-augmented structures initialize new primary geometry.
- Cached actual-target CUDA thread/shared-memory checks are lowered to a
  constant before native execution. Portable lowering without a definite
  assignment conservatively checks all local backend candidates.
- JAX 0.11.0 focused CPU/CUDA suite: 279 passed. Optimized HLO confirms actual
  combined calls; the suite does not infer dispatch from matching values.
- JAX 0.9.0 CPU suite: 232 passed, 47 GPU cases skipped. GPU execution on
  JAX 0.9.0 was not verified in this environment.
- Relevant implementation coverage: 96.49% overall; main 94.15%, geometry
  preprocessing 98.25%, dispatch 95.65%, resources 97.62%, cache 100%.
- Seven source files pass focused mypy. Patch-format checks and public API
  doctests pass. CSR/CSC/FixedNum cache regressions: 80 passed, 4 skipped.
- Single-GPU Compute Sanitizer: two tail-tile/vmap tests passed, process exit
  code 0 and ERROR SUMMARY: 0 errors (`/tmp/tcsr-auto-memcheck.log`).
- Development documents and the old implementation were moved under
  `_tcsr/dev` during implementation; documentation follows that location.
