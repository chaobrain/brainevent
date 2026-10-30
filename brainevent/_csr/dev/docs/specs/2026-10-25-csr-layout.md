# CSR functional directory layout

## Scope

Implement the approved [CSR layout and dependency plan](../../csr-structure-plan.md)
on the local `csr-layout` branch, based on `main` at `f0f93b1` (0.3.0).
Keep the downloaded workspace changes present before the migration. Do not push.

Group complete existing files under `binary`, `float`, `delay`, `slice`, `dt2t`,
`plasticity`, `diag`, and `solve`; put `csr_to_csc.cu` under `_support`.
Keep `main.py`, the root exports, and shared test helpers at the CSR root.
Group each functional area's existing tests under its `test/` subdirectory,
following the [updated test layout](2026-10-25-csr-test-layout.md).

## Behavior and compatibility

- Preserve algorithms, signatures, primitive names, backend registration, FFI names,
  CUDA source contents, and caller/callee relationships.
- Retarget imports to the moved implementations, including lazy imports,
  monkeypatch targets, and subprocess test imports.
- Keep the `brainevent` and `brainevent._csr` public exports unchanged.
  New functional package exports reference the implementation objects directly.
- Private module paths that move to a different functional package use their new
  canonical paths. Document the mapping; do not introduce module alias machinery.
- Preserve binary-to-float differentiation and plasticity-to-binary helper reuse.
- Keep SDDMM in `brainevent/_sddmm.py`; leave TCSR and existing CSR experiments alone.
- Keep workspace management in `main.py` and structural utilities in `_misc.py`.
  Only retarget `_misc.py`'s CUDA resource path to `_support/csr_to_csc.cu`.

## Validation

- Compare moved CUDA files byte for byte with the pre-migration snapshot.
- Check that implementation ASTs differ only in imports and documented paths.
- Run existing CSR tests and affected caller/structure utility tests on CPU.
- Verify public exports and cross-function imports, including derivative paths.
- Build a wheel and check installed imports and CUDA resource inclusion/location.
- Run the repository mypy gate in isolation and compare any failures to baseline.

Edge cases include lazy delay imports, indexed binary reuse, module monkeypatches,
CSR/CSC structural conversion, wheel resource discovery, and recursive CUDA source
checks excluding the unchanged `dev` experiments. GPU execution requires a GPU
validation environment; source relocation alone does not verify kernel execution.

## Result

Implemented the file mapping, package exports, import paths, shared CUDA resource
path, and recursive source checks. The existing lazy-loader test now also verifies
that the resource passed to the loader exists.

CPU regression: 636 passed, 79 skipped, 2080 deselected with the default `not slow`
selection (Python 3.12, JAX 0.12.0.1). Mypy with `--no-site-packages` passes all
107 source files; the same baseline gate passes 99 files. Wheel build, installed
imports, all 17 CUDA resource paths, public exports, and 57 primitive registrations
were verified. All CUDA files are byte-identical to the original local snapshot.
GPU execution was not tested. Detailed results are in the linked CSR plan.
