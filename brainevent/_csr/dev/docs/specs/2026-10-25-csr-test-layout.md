# CSR functional test directories

## Scope

Follow the user's updated layout requirement: move every existing formal CSR test
module into the `test/` subdirectory of its functional area. Root-level object
and shared-helper tests belong in `brainevent/_csr/test/`. Keep `*_test.py` names
and leave the shared `_test_util.py` implementation at the CSR root.

This supersedes the sibling-test layout for CSR only. Existing `dev/` experiments
and other packages are outside this change. Keep all computation and dispatch
implementations unchanged. Work only on local `csr-layout`; do not push.

## Implementation

- Move 14 existing test modules into eight `test/` packages: CSR root, binary,
  float, delay, slice, dt2t, plasticity, and diag. No solve test currently exists.
- Add package initializers so pytest uses qualified module names and avoids
  collisions with similarly named tests in other functional areas/packages.
- Update delay-layout tests' import of delay test helpers.
- Fix the two source scans that locate CUDA files relative to the test's own
  file: the CSR-wide invariant and binary hybrid ABI-name test.
- Update the current CSR structure documentation and scoped repository rule.

## Validation

Compare collected test node IDs before and after, allowing only mapped file
paths to differ. Run the existing CSR CPU regression, including source-location
and shared-helper tests. Check imports and mypy, and verify byte-identical
computation/dispatch sources. GPU kernels are not changed by this move.

## Result

All 14 test modules now live in eight `test/` packages. All 2503 collected CSR
cases match the pre-migration collection after mapping paths; no cases were lost
or duplicated. Eleven test files were moved unchanged; three required only a
helper-import or source-location adjustment. Existing production Python/CUDA
sources are byte-identical to their pre-move versions.

CPU regression (`JAX_PLATFORMS=cpu`, default `not slow` selection): 499 passed,
79 skipped, 1925 deselected. The two warnings come from the existing explicit
module-reload test. `mypy --no-site-packages` passes 116 source files. GPU execution
and deselected slow variants were not run.
