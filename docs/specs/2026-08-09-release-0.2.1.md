# Release 0.2.1 — Daily CI repair, API reference refresh

Status: implemented
Branch: `worktree-release-0.2.1`

## 1. Motivation

Three independent problems were bundled into the 0.2.1 patch release:

1. **Daily CI Tests failed** on the pinned-JAX matrix legs (runs `31234230842`,
   `31290069801`) while the unpinned leg passed.
2. **The Daily CI matrix did not cover JAX 0.10.x**, the newest supported minor,
   so the release matrix silently under-tested the version most users install.
3. **`docs/reference/apis/` had drifted** from the public API. Symbols removed in
   0.2.0 were still documented, and 44 public exports had no reference page.

## 2. Daily CI failure — root cause

Failing test: `brainevent/_op/main_test.py::test_f17_kernel_generator_failure_is_wrapped_with_alternatives`.

The test asserted that a kernel-generator exception is preserved as the direct
`__cause__` of the raised `KernelCompilationError`:

```python
assert isinstance(excinfo.value.__cause__, RuntimeError)
```

JAX's `jax._src.traceback_util.api_boundary` rewrites the cause chain of any
exception that crosses an API boundary:

```python
jax_error.__cause__ = e.__cause__      # original cause is pushed one level down
e.__cause__ = jax_error                # synthetic frame becomes the direct cause
```

The synthetic frame is `UnfilteredStackTrace` (JAX 0.10) or
`JaxStackTraceBeforeTransformation` (JAX 0.8/0.9). Whether it is inserted depends
on the active `jax_traceback_filtering` mode, whose default differs across JAX
minors — hence the failure on pinned 0.8.0/0.9.0 and the pass on the unpinned leg.

The library behaviour was correct throughout; only the assertion was too strict.

### Fix

Landed in `a66dcee`: assert against the whole `__cause__` chain rather than the
first link.

```python
def _exception_chain_contains(exc, expected_type, expected_text: str) -> bool:
    while exc is not None:
        if isinstance(exc, expected_type) and expected_text in str(exc):
            return True
        exc = getattr(exc, '__cause__', None)
    return False
```

This is invariant to how many synthetic frames JAX splices in, so it holds for
every supported JAX version.

### Verification

Clean virtualenvs pinned to each matrix version, full suite (`pytest brainevent/ -m ""`):

| JAX | Result |
| --- | --- |
| 0.8.0 | pass |
| 0.9.0 | pass |
| 0.10.2 (unpinned) | pass |

## 3. Daily CI matrix

`jax-version` extended from `[ "0.8.0", "0.9.0", "" ]` to
`[ "0.8.0", "0.9.0", "0.10.0", "" ]`, so every supported minor is pinned
explicitly and the empty entry continues to track the newest release.

## 4. API reference refresh

`docs/reference/apis/*.rst` was audited by comparing every `autosummary` entry
against `brainevent.__all__` and `brainevent.config.__all__`.

### 4.1 Stale entries removed (41)

| File | Removed |
| --- | --- |
| `events.rst` | `IndexedEventRepresentation`, `IndexedBinary1d`, `IndexedBinary2d` |
| `operations.rst` | `csr_solve`, `indexed_binary_densemv{,_p}`, `indexed_binary_densemm{,_p}` |
| `utilities.rst` | `binary_array_index`, `BenchmarkReport`, `register_cuda_kernels`, and the 24 `lfsr88_*` / `lfsr113_*` / `lfsr128_*` plus 6 `get_numba_*` entries |

The `lfsr*` and `get_numba_*` helpers live in the private `brainevent._numba_random`
module and are not re-exported, so `currentmodule:: brainevent` could never resolve
them; they are kernel-generation internals and are intentionally not part of the
public reference.

### 4.2 Public exports added (44)

| File | Added |
| --- | --- |
| `events.rst` | `BitPackedBinary`, `CompactBinary`, `bitpack` |
| `sparsedata.rst` | `Dense`, `JITCScalarMatrix`, `FixedNumPerPre`, `FixedNumPerPost` |
| `operations.rst` | `binary_csrmv_indexed{,_p}`, `binary_csrmm_indexed{,_p}`, `update_csc_on_binary_pre`, `update_csc_on_binary_post`, `fcn_plasticity_row_p`, `update_fixed_pre_conn_on_binary_post`, `update_fixed_post_conn_on_binary_pre` |
| `operator.rst` | `CompilerBackend`, `CPPBackend`, `CUDABackend`, `HIPBackend`, `normalize_tokens`, `get_registry`, `get_all_primitive_names`, `get_primitives_by_tags` |
| `utilities.rst` | `csc_to_csr_index`, `BenchmarkConfig`, `BenchmarkRecord`, `HybridConfig`, `get_hybrid_config`, `init_csr_config` |
| `errors.rst` | `KernelError`, `KernelLoadError`, `KernelRegistrationError`, `KernelToolchainError`, `NvccNotFoundError`, `HeaderNotFoundError`, `HostCompilerNotFoundError`, `HostCompilerIncompatibleError`, `GpuArchDetectionError`, `UnsupportedArchError`, `CompilationError`, `BrainEventError`, `UnsupportedOperationError`, `BenchmarkDataFnNotProvidedError` |
| `config.rst` | `get_compute_capability`, `set_compute_capability`, `prefer_system_nvcc` |

### 4.3 Structural changes

- **Exceptions consolidated.** `operator.rst` previously documented four exception
  classes while `errors.rst` documented six others, leaving ten undocumented.
  All twenty now live in `errors.rst`, grouped by subtree and preceded by the
  inheritance tree; `operator.rst` cross-references it.
- **Deprecated aliases dropped.** `FixedPreNumConn` / `FixedPostNumConn` emit a
  `DeprecationWarning` on attribute access; the reference now documents the
  replacements `FixedNumPerPre` / `FixedNumPerPost`.

### 4.4 Prose pages

The narrative docs cited the same retired symbols, and two of their snippets no
longer ran at all:

| Page | Problem | Fix |
| --- | --- | --- |
| `getting-started/quickstart.rst` | `JITCScalarR(num_pre=…, num_post=…, prob=…, weight=…, seed=…)` and `FixedPostNumConn(num_pre=…, conn_num=…, weight=…, seed=…)` — neither signature exists; both raise `TypeError`. `csr` was built from undefined `data` / `indices` / `indptr`. | rewritten against the real constructors and executed end to end |
| `how-to/building-extending/build-coba-cuba-network.rst` | same non-existent `FixedPostNumConn` signature | rewritten with explicit `(data, indices)` and unit-carrying weights; executed end to end |
| `explanation/event-driven-computation.rst` | cites `IndexedBinary1d` / `IndexedBinary2d` (removed) | replaced by `BitPackedBinary` / `CompactBinary` |
| `explanation/faq.rst`, `explanation/sparse-formats.rst`, `how-to/data-structures/choosing-a-sparse-format.rst` | deprecated `Fixed*NumConn` aliases | `FixedNumPerPre` / `FixedNumPerPost` |
| `tutorials/data-structures/03_jit_connectivity.ipynb`, `04_fixed_connections.ipynb` | 43 uses of the deprecated aliases across prose, code and recorded output | renamed; notebook JSON re-validated |

The alias mapping is crossed, so the rename is not a search-and-replace of the
obvious pairs: `FixedPostNumConn` → `FixedNumPerPre` (a fixed number of *post*
-synaptic connections per pre-synaptic neuron) and `FixedPreNumConn` →
`FixedNumPerPost`. Both directions were taken from `brainevent/_deprecation.py`
rather than inferred from the names.

### 4.5 Regression guard

`docs/reference/apis/api_coverage_test.py` re-runs the audit as a test: every
`autosummary` entry must resolve on its `currentmodule`, and every name in
`brainevent.__all__` must appear in exactly one reference page. Deprecated
aliases and the `config` submodule handle are the only allowed exemptions.

## 5. Edge cases and test coverage

| Edge case | Handling |
| --- | --- |
| JAX splices 0, 1 or N synthetic frames into `__cause__` | chain walk terminates on `None`; type + message both checked |
| Cause chain cycles | not reachable — Python forbids self-referential `__cause__` assignment loops in this path; the chain is built strictly downward by `api_boundary` |
| A new public export added without a doc entry | `api_coverage_test.py` fails |
| A doc entry outliving its symbol | `api_coverage_test.py` fails |
| Deprecated alias re-added to a reference page | allowed-exemption list is explicit, so it must be edited deliberately |
| `brainevent.config` documented under its own `currentmodule` | test resolves `config.rst` entries against `brainevent.config` |
