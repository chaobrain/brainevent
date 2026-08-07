# Operator-registration hardening

Status: implemented (branch `worktree-fix-op-registration-audit`).

## Problem

An audit of the custom-operator registration machinery under `brainevent/_op`
catalogued 19 defects (F1–F19). They share a theme: the registration and
dispatch layer failed *silently*. A wrong backend, a dropped gradient, a stale
compiled artefact, or a mis-batched CUDA launch produced a plausible-looking
number rather than an error, so none of them were visible from the test suite's
pass/fail signal alone.

The findings cluster into five groups.

### F1 — backend switches did not take effect

`XLACustomKernel.set_default`, `brainevent.config.set_backend` and
`clear_backends` mutated brainevent's own registry but left JAX's dispatch and
executable caches untouched. Eager calls and already-traced `jax.jit` functions
kept running the *previous* backend, so a user switching from `cuda_raw` to
`jax_raw` to isolate a numerical discrepancy would measure the same kernel
twice.

### F2 — `defjvp` silently dropped JVP rules

`defjvp` paired rules with tangents via `zip(jvp_rules, tangents)`. `zip` stops
at the shorter sequence, so registering fewer rules than the primitive has
inputs dropped the trailing inputs' tangent contributions with no error — an
incorrect gradient, silently. The multi-result path additionally accepted a
bare array where a sequence was required and fragmented it by iterating its
leading axis.

One in-tree instance existed: `binary_fcnmm_p` registered four rules for a
three-input primitive.

### F3, F4, F10, F12 — incomplete compilation-cache keys

The kernix (inline C++/CUDA) cache key hashed only the user's source text. It
therefore ignored:

- the resolved `FunctionSpec`s (F3) — loading the same source first with one
  function and then with two hit the single-function `.so`, which lacks the
  second wrapper symbol, and failed with a misleading "Did the compilation
  succeed?" message;
- the content of headers reachable through `extra_include_paths` (F10) —
  editing such a header was invisible to the cache.

Separately, re-registering a changed artefact under an existing FFI target name
(F4, F12) had no well-defined behaviour: `force_rebuild=True` and `replace=True`
could leave the *old* handler installed and keep dispatching stale code.

### F5, F9 — `vmap` over `numba.cuda` kernels computed wrong results

Batched calls reused the launch grid computed for the *unbatched* shape and ran
it once over the folded buffer. Any kernel that couples rows — stencils,
reductions, segment scans — silently produced corrupt output for all but the
first slice. Nested `vmap` returned uninitialized memory.

### F6–F8, F11, F13–F19 — diagnostics and identity

FFI target names were assigned from a per-process counter, so `jax.export`
artefacts were not stable across processes and a recycled Python object `id`
could mis-dispatch. Failure modes surfaced as bare `assert`s, `KeyError`s, or
silent fallbacks rather than typed, actionable errors.

## Approach

**Invalidate on effective change, not on every call.** Backend setters compare
the resolved setting before and after; only a real change triggers
`jax.clear_caches()`. The invalidation is process-global — the next call of
every jitted function recompiles — which is the correct trade for a setting
that is changed rarely and interactively.

**Fail loudly at the earliest point where the mistake is still attributable.**
`defjvp` checks rule/input arity at registration-time semantics (before the
`zip`) and raises `ValueError` naming the primitive; a multi-result rule
returning a bare array raises `TypeError`. Unknown, packed sub-byte
(`S1`–`S4`, `U1`–`U4`, `F4E2M1FN`) and FP8 buffer dtypes raise instead of being
reinterpreted as raw bytes.

**Make registration identity deterministic.** The cache key (schema v2) covers
the resolved `FunctionSpec`s and the content of extra include headers; the
deterministic cache key *is* the registration identity. Two consequences follow
directly:

- re-registering byte-identical content (e.g. `force_rebuild=True` twice) is an
  idempotent no-op, not an error;
- re-registering *changed* content under an existing target name raises
  `KernelRegistrationError` on every platform.

The second is a deliberate behaviour change. Live re-pointing of an
already-registered XLA FFI target is not supported by the JAX versions this
package targets — probed directly: the Host registry rejects a differing bundle
address, and the CUDA registry accepts the call but silently keeps the old
handler. Since a correct re-point cannot be *verified*, brainevent refuses
deterministically and the error names both remedies (`replace=True`, or a
distinct `name=` / `target_prefix=`). Refusing is strictly better than the
pre-fix behaviour of appearing to succeed while dispatching stale code.

**Derive FFI target names from content.** Names are a fingerprint of the
kernel's bytecode, constants, closure values and referenced globals
(dispatcher-aware), which makes `jax.export` artefacts stable across processes.
Kernels whose content cannot be fingerprinted deterministically fall back to
per-process counter names — the pre-fix behaviour, now the exception rather than
the rule. Reuse branches pin the kernel object so a recycled `id` cannot
mis-dispatch, and handlers self-pin at construction.

**Batch `numba.cuda` by launching per slice.** Rank-based detection identifies
the batched case and issues one launch per batch slice with the kernel's
original launch configuration. Kernels wrapped with an explicit `grid=` cannot
be batched (combining `grid=` with `vmap_method=` raises `ValueError` at wrap
time); nested `vmap` raises rather than returning garbage. CUDA output buffers
for accumulating kernels are zero-filled on XLA's stream.

## Behaviour changes

Two changes are visible to users and are recorded in `changelog.md`:

1. Re-registering an FFI target with *different* content raises
   `KernelRegistrationError` on every platform, including
   `load_cuda_inline(..., replace=True)` and `force_rebuild=True` with edited
   source. To iterate on a kernel within one process, register under a new
   `name=`.
2. Kernel construction/compile failures during lowering raise
   `KernelCompilationError`, with the original exception as `__cause__` and the
   remaining registered backends listed. Calling a kernel on a platform with no
   registered backend raises `KernelFallbackExhaustedError`. Both are exported
   from `brainevent`.

Change 2 has one in-tree consequence: the removed CUDA row-gather matvec raised
a bare `NotImplementedError` from its construction path, and
`_fcn/binary_test.py::test_ell_binary_matvec_forward_matches_reference` asserted
on that exact type. The message (and thus the `match='row-gather'` assertion) is
preserved inside the wrapper, so the test now accepts either.

Registering a second primitive under an existing name emits a `UserWarning`; the
new registration still wins, as before.

## Verification

Every finding carries a regression test that fails against the pre-fix code:

| Area | Test module |
|---|---|
| Backend-switch invalidation (F1) | `brainevent/config_test.py`, `brainevent/_op/main_test.py` |
| `defjvp` arity (F2) | `brainevent/_op/util_test.py` |
| Cache keys, re-registration (F3, F4, F10, F12) | `brainevent/_op/kernix_cache_test.py`, `brainevent/_op/kernix_pipeline_test.py` |
| `numba.cuda` vmap (F5, F9) | `brainevent/_op/numba_cuda_ffi_test.py` |
| Naming, typed errors (F6–F8, F11, F13–F19) | `brainevent/_op/numba_ffi_test.py`, `brainevent/_op/kernix_runtime_test.py` |

The F3/F4/F10/F12 end-to-end tests exercise *cache-hit* behaviour, so each takes
an `isolated_cache` fixture pointing the compilation cache at a scratch
directory. Without it, another test in the session can pre-populate the cache
and the test passes for the wrong reason.

The F1/F17 probes register kernels for `'cpu'` only. Their input arrays must be
committed with `jax.device_put(..., jax.devices('cpu')[0])`: on a machine with a
GPU, default placement lowers the primitive for `'gpu'` and the call dies with
`KernelFallbackExhaustedError` before the behaviour under test is reached. These
tests were originally written against a CPU-only run and passed for that reason;
a full run on CUDA hardware is what exposed the omission.

CPU-only runs cannot cover the CUDA paths — every CUDA test is gated behind
`requires_gpu` and `.cu`/inline-CUDA sources are never compiled unless a GPU
kernel is lowered. Validation therefore requires a run on CUDA hardware in
addition to the CPU suite and the `mypy` gate.

## Known gaps

- `brainevent/_op/ffi_naming.py` has no sibling `ffi_naming_test.py`. Its
  behaviour is exercised indirectly through `numba_ffi_test.py` and
  `numba_cuda_ffi_test.py`, which is adequate coverage but does not satisfy the
  co-location rule. Splitting out a direct test module is a follow-up.
- The audit's own working notes (`dev/2026-07-16-op-registration-audit.md`,
  referenced by the implementing commit message) were never committed; this
  document supersedes them.
