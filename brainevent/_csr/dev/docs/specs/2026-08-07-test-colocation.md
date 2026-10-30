# Test co-location spec

Status: implemented
Branch: `worktree-test-colocation-rule`

## Goal

Make every test module in `brainevent/` satisfy `AGENTS.md` rule 11:

> Co-locate tests with the code under test: each module `foo.py` has its tests in a sibling
> `foo_test.py` (suffix style — never a separate `tests/` directory, never the `test_*.py` prefix).

Two machine-checkable invariants define "done":

1. `find brainevent -name 'test_*.py'` prints nothing.
2. For every `brainevent/**/foo_test.py` there is a sibling `brainevent/**/foo.py`.

There is no `tests/` directory in the repository and every test module already uses the `_test.py`
suffix, so the third clause of the rule already holds and stays holding.

## Starting state

Baseline on `main` (commit `2b9a91d`): **3636 tests collected** with `pytest --collect-only -m ''`.

Invariant 1 is violated by one file. Invariant 2 is violated by eleven.

| Orphan test file | Tests | Code it actually exercises |
| --- | ---: | --- |
| `_data_contract_test.py` | 191 | `DataRepresentation` in `_data.py` |
| `deprecation_test.py` | 45 | `__getattr__` shim in `__init__.py` |
| `_csr/cuda_int64_indptr_test.py` | 43 | int64-indptr policy across 7 `_csr` modules |
| `_fcn/conversion_test.py` | 42 | `FixedNumConn.todense/tocsr/tocsc` in `_fcn/main.py` |
| `_csr/indptr_dtype_test.py` | 38 | `_misc.py` dtype resolvers, CSR ctor, `diag_add` |
| `_op/kernix_dtypes_test.py` | 29 | `load_cuda_inline` dtype matrix (`kernix_pipeline.py`) |
| `_fcn/slice_test.py` | 18 | `FixedNumPer*.__getitem__` / `slice_rows` in `_fcn/main.py` |
| `_csr/binary_workspace_test.py` | 9 | `_BinaryTaskWorkspace` &co. in `_csr/main.py` |
| `_op/kernix_cuda_test.py` | 5 | `load_cuda_inline` (`kernix_pipeline.py:155`) |
| `_op/kernix_cpp_test.py` | 5 | `load_cpp_inline` (`kernix_pipeline.py:390`) |
| `_csr/binary_backend_test.py` | 2 | `binary_csrmv_p` / `binary_csrmm_p` in `_csr/binary.py` |

Prefix violation: `_csr/test_util.py` (helper, 20 tests in its `test_util_test.py`).

## Design decisions

### D1 — `deprecation_test.py` has no legal sibling; extract the shim

`__init__.py` cannot have a sibling test under the rule (`__init___test.py` is legal but unreadable).
The shim is self-contained data plus one resolver, so it moves to a real module
`brainevent/_deprecation.py` and the test becomes `_deprecation_test.py`.

`_DEPRECATED_RENAMES` currently maps old names to **live objects** (`BinaryArray`, …). Importing
those into `_deprecation.py` would be circular, so the table stores **target name strings** and
resolution happens against the caller's namespace:

```python
def resolve(name, namespace)      # warns + returns namespace[target], or raises AttributeError
def public_dir(namespace)         # sorted(namespace | renames | removed)
```

This is the only shipping-code change in the whole task. It is behaviour-preserving: the public
surface (`brainevent.EventArray` warns and forwards; `brainevent.COO` raises with a migration
message; `dir(brainevent)` lists both) is unchanged, and it gains testability — `resolve` can now be
unit-tested against a synthetic namespace.

### D2 — cross-cutting suites are split by target module, not relocated whole

`cuda_int64_indptr_test.py` and `indptr_dtype_test.py` are organised by *policy* (int64 indptr
support) rather than by module. Two of their tests are parametrized across modules:

- `test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda` — 9 params spanning float,
  binary, binary_indexed, slice, dt2t.
- `test_slice_dt2t_and_plasticity_cuda_generators_accept_int64_indptr_without_real_cuda`.

These are **split into per-module copies** carrying only their own module's parameters. This renames
node IDs, so exact node-ID parity does not hold for these two tests — see "Verification" for how the
count is reconciled instead.

Shared fixtures (`_structure`, `_shape`, `_cuda_kwargs`, `_strip_hybrid_suffix`, `_small_csr`) move
to `_csr/_test_util.py`. `_jax_x64_enabled` is needed on both sides of the `_misc` / `_csr` boundary,
so it goes to the package-wide `brainevent/_test_util.py` instead of being duplicated.

### D2b — a merge must not silently change a test's marker scope

Found during implementation, not anticipated in the plan. Three merge destinations carried a
*module-level* `pytestmark = pytest.mark.slow`: `_csr/binary_test.py`, `_csr/main_test.py`,
`_fcn/main_test.py`. Appending fast tests to those files would have pushed 152 currently-fast tests
out of the default `pytest` run — invisible to a collected-count check, because they still collect
under `-m ''`. One merged file's own docstring explicitly stated it "is *not* marked ``slow`` and
runs in the default ``pytest`` lane", so the regression would have contradicted a written promise.

Resolution: the blanket `pytestmark` is replaced by explicit `@pytest.mark.slow` on each top-level
item that was previously slow (20 / 29 / 22 items respectively), applied mechanically via an `ast`
pass and verified to leave the fast/deselected split unchanged before and after.

The `_op` merge had two analogous module-scope hazards, handled the same way:

- `kernix_cpp_test.py` used `pytest.skip(allow_module_level=True)` on Windows — that would have
  skipped the whole merged file. It becomes a per-test `_skip_on_windows` marker.
- `kernix_dtypes_test.py` had a module-level `requires_gpu` plus an `autouse` x64 fixture that never
  restored the flag. Both are confined inside a new `class TestDtypes`, which renames those 29 node
  IDs (documented in "Verification").

General rule this encodes: when merging, module-scope state (`pytestmark`, module-level `skip`,
`autouse` fixtures) must be narrowed to the tests that owned it, never inherited by the destination's
existing tests.

### D3 — two orphan tests become the first tests for untested modules

`diag_add.py` and `hybrid_config.py` have no test file today. The `diag_add` gating tests and the
hybrid-ABI-naming test extracted in D2 land there as `_csr/diag_add_test.py` and
`_csr/hybrid_config_test.py`, satisfying the rule and closing two coverage gaps as a side effect.

## Destination map

Straight merges (append, drop duplicate license header, fold duplicate imports):

| From | Into |
| --- | --- |
| `_data_contract_test.py` | `_data_test.py` |
| `_csr/binary_backend_test.py` | `_csr/binary_test.py` |
| `_csr/binary_workspace_test.py` | `_csr/main_test.py` |
| `_fcn/conversion_test.py`, `_fcn/slice_test.py` | `_fcn/main_test.py` |
| `_op/kernix_{cpp,cuda,dtypes}_test.py` | `_op/kernix_pipeline_test.py` |

`indptr_dtype_test.py` splits three ways: `_misc` helpers → `brainevent/_misc_test.py`; CSR/CSC
constructor and methods → `_csr/main_test.py`; `diag_add` gating → new `_csr/diag_add_test.py`.

`cuda_int64_indptr_test.py` splits eight ways across `_csr/{binary,binary_indexed,float,slice,dt2t,
plasticity_binary,main}_test.py` plus new `_csr/hybrid_config_test.py`.

Rename: `_csr/test_util.py` → `_csr/_test_util.py`, `_csr/test_util_test.py` →
`_csr/_test_util_test.py`, plus five importers and the `[tool.coverage.run] omit` glob in
`pyproject.toml`.

## Out of scope

Coverage gaps, not naming violations — recorded so the omission is deliberate, not accidental:

- No test file at all: `_typing.py`, `_version.py`, `_op/benchmark.py`, `_csr/spsolve.py`,
  `_event/bitpack_binary.py`.
- `_jit_normal/_test_util.py` and `_jit_uniform/_test_util.py` are test helpers with no test file.
  They already use the compliant `_` prefix; untested helper modules match existing convention.

No test body is rewritten to change what it asserts. Merges are mechanical; the only edits to test
logic are the parameter-list narrowing required by the D2 splits.

## Verification

1. Both invariants, as greps (see "Goal").
2. Collected-test reconciliation against the 3636 baseline. The D2 splits change node IDs for two
   tests, so parity is checked as: total count unchanged, and the set difference of normalized node
   IDs is empty except for the documented split renames.
3. `pytest -q` (default, `-m 'not slow'`) and `pytest -q -m ''` (full, as CI runs it).
4. Deprecation-shim behaviour probe — warn-and-forward, removed-name `AttributeError`, `__dir__`
   contents — since D1 is the only shipping-code change.
5. `mypy` (the `exclude = ['_test\.py$']` pattern must still match every renamed file) and a coverage
   run confirming `_csr/_test_util.py` is not counted as library code.

## Result

Both invariants hold: no `test_*.py`, no `tests/` directory, no orphan `*_test.py`.

Collection went from **3636 / 1775** (full / fast) to **3648 / 1787** — `+12` in both lanes. The
normalized node-ID diff is 37 removed against 49 added, each entry accounted for:

| Δ | Count | Cause |
| --- | ---: | --- |
| renamed | 29 | `_op` dtype tests nested under `TestDtypes` (D2b) |
| renamed | 7 | reject-generator params renumbered by their new per-file param lists (D2) |
| split | 1 → 3 | `test_slice_dt2t_and_plasticity_..._without_real_cuda` (D2), net `+2` |
| added | 10 | new direct unit tests for `_deprecation.resolve` / `public_dir` (D1) |

`pytest -q` → 1586 passed, 201 skipped, 1861 deselected. The fast-lane count is `+12` over baseline,
i.e. the D2b marker rework moved nothing between lanes.
`pytest -q -m ''` → 3426 passed, 222 skipped, 0 failed.

The `[tool.coverage.run] omit` globs still cover all four `_test_util.py` helpers, including the
renamed `_csr/_test_util.py` that the old `brainevent/**/test_util.py` glob would have stopped
matching.

`mypy` reports no error in `_deprecation.py` or `__init__.py`; the 204 errors it does report are all
in library modules this change does not touch and are pre-existing.
