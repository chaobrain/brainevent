# FCN Branch Review

Branch: `fcnmv-column-major-operator`

This note summarizes the FCN-related changes currently present on the branch, with emphasis on what changed in behavior, what correctness bugs were fixed, and what still remains partial or unfinished.

## Confirmed major changes

### 1. Bitpack MM pack-axis propagation and validation

This item is present and is broader than a simple parameter addition.

- `bitpack_binary_fcnmv()` now accepts `mm_pack_axis` and `backend`, and forwards them into the primitive call path when MV is promoted to MM by batching.
- `_bitpack_binary_fcnmv_batching()` now resolves the FCN-MM backend explicitly and chooses the promoted MM packing layout based on `mm_pack_axis`.
- `bitpack_binary_fcnmm()` / `bitpack_binary_fcnmm_p_call()` now also accept `backend`, so the promoted MM path can keep the intended backend instead of silently falling back.
- `FixedPostNumConn` and `FixedPreNumConn` now propagate both `bitpack_mm_pack_axis` and the resolved MM backend through all bitpack MV/MM matmul entrypoints.
- `main.py` now validates `bitpack_mm_pack_axis` strictly: booleans and non-integers are rejected, not just values outside `{0, 1}`.

Files:
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:39)
- [main.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/main.py:125)
- [main_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/main_test.py:280)
- [bitpack_binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary_test.py:814)

### 2. MV column-scatter formal integration

This item is present and is the main architectural change of the branch.

- The old `binary_fcnmv_T.cu` path is gone.
- A new dedicated CUDA source `binary_fcnmv_col_scatter.cu` is introduced and loaded under `fcn_binary_mv_col_scatter`.
- `binary_fcnmv` now treats `transpose=False` as requiring an explicit column-major mirror (`col_weights`, `col_indices`, `col_indptr`); without that mirror it raises a `ValueError` and points users to column-scatter or bitpack alternatives.
- Non-CUDA backends and post/scatter calls with a mirror no longer pretend to use the mirror; they warn and fall back to the regular path.
- Dual-layout maintenance in `Fixed*Conn` and the COBA benchmark wiring now support this route intentionally instead of incidentally.

Files:
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:473)
- [binary_fcnmv_col_scatter.cu](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary_fcnmv_col_scatter.cu:1)
- [binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary_test.py:108)

### 3. Removal of old low-performance operators

This item is present, and it covers more than one operator family.

- The old compact FCN operator implementations under `_fcn` were effectively removed.
  - `compact_binary_fcnmv.cu`
  - `compact_binary_fcnmv_T.cu`
  - `compact_binary_fcnmm.cu`
- `compact_binary.py` in `_fcn` is now a compatibility stub that raises a removed-operator error instead of dispatching real kernels.
- The old dummy backend path was removed.
  - `dummy.cu` deleted
  - `dummy_backend_test.py` deleted
  - `dummy_kernel` unregistered from both binary and bitpack MV primitives
- Bitpack MV scatter (`transpose=True`) is now explicitly rejected in CUDA instead of remaining as a slow/incorrect path.

Files:
- [compact_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/compact_binary.py:1)
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:113)
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:853)

### 4. Batching logic and parameter propagation updates

This item is definitely present.

- `binary_fcnmv` batching now promotes batched MV calls to `binary_fcnmm_p_call()`, and restores public layout when the selected MM backend uses a non-public internal layout.
- `bitpack_binary_fcnmv` batching now:
  - resolves the FCN-MM backend explicitly,
  - repacks the matrix differently for `a0` vs `a1`,
  - passes `pack_axis`,
  - passes `backend`.
- `bitpack_binary_fcnmm` batching now propagates `backend` through the batching base path and fallback rule.

Files:
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:691)
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:313)

### 5. Explicit `backend` is now a strong parameter

This should be recorded as a separate item.

- FCN call sites now more consistently accept and forward `backend` as an explicit runtime parameter.
- When `backend` is explicitly passed, it is treated as a strong override, not a preference.
- In that case the dispatch path does not fall back to the global backend.
- If the requested backend is not registered for the current primitive/platform, dispatch fails directly instead of silently choosing another backend.

This is visible in two layers:

- FCN helpers first prefer the explicit argument.
- The primitive dispatch layer raises `KernelFallbackExhaustedError` if that explicit backend is unavailable for the current platform/primitive.

Files:
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:872)
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:314)
- [main.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/main.py:139)
- [main.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_op/main.py:393)

## Additional changes beyond the listed items

These are also substantial branch changes and should be recorded.

### 6. `test_colmajor_fullwarp_nocap` was added as a `binary_fcnmm` backend

- `binary_fcnmm_p` now registers a GPU backend named `test_colmajor_fullwarp_nocap`.
- It uses `fcnmm_testing_op.cu` through `_binary_fcnmm_test_colmajor_kernel()`.
- Public layout restoration was added so this backend can return normal user-facing tensor layout even though the internal kernel uses a transposed/colmajor-oriented output contract.
- There is a dedicated primitive-level pytest suite that verifies:
  - registration,
  - kernel name wiring,
  - correctness vs reference,
  - participation when selected as the global backend,
  - participation through the batched `binary_fcnmv` -> `binary_fcnmm` promotion path.

Files:
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:1172)
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:1474)
- [binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary_test.py:699)

### 7. Event/compact preprocessing got new `jax_raw` fallback kernels

- `_event/compact.py` now adds `jax_raw` kernels for multiple preprocessing primitives that previously relied on CUDA or Numba.
- A new dense-to-CSC encoder was added in pure JAX.
- `CompactBinary.from_array()` and related constructors now choose preprocessing backends more deliberately.

This matters because some earlier compact preprocessing assumptions were backend-fragile and hard to test under `jit`/`vmap`.

Files:
- [compact.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_event/compact.py:198)
- [compact_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_event/compact_binary.py:32)

### 8. COBA benchmark/test workflow was rebuilt around explicit MV/MM route coverage

- The old monolithic `coba_ei_benchmark_test.py` was split into:
  - `coba_ei_benchmark_mv_test.py`
  - `coba_ei_benchmark_mm_test.py`
  - `coba_ei_benchmark_test_helpers.py`
- COBA benchmark constructors now accept `backend`.
- MM test coverage was narrowed so removed routes are no longer treated as successful benchmark routes.
- Intermediate tracing helpers were added for large-scale MM/MV correctness checks.

Files:
- [COBA EI benchmark.py](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/COBA%20EI%20benchmark.py:72)
- [coba_ei_benchmark_test_helpers.py](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_benchmark_test_helpers.py:1259)
- [coba_ei_benchmark_mv_test.py](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_benchmark_mv_test.py:1)
- [coba_ei_benchmark_mm_test.py](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_benchmark_mm_test.py:1)

## Basic correctness fixes

This section records behavior that was wrong before the branch changes, and what is fixed now.

### A. Bitpack MV batching previously lost MM layout/backend intent

Before:
- When `bitpack_binary_fcnmv` was batched into MM, the promoted path effectively assumed pack-axis 0 and did not faithfully carry the intended MM backend through the batching rule.
- `bitpack_a1` could therefore be promoted with the wrong packed layout.
- Some tests compared one kernel call against another call to the same kernel, which could hide a shared mistake.

After:
- `mm_pack_axis` is explicitly threaded through MV, primitive call, batching rule, `Fixed*Conn`, and MM primitive call.
- The promoted MM path now repacks for `a1` correctly and passes the selected backend.
- Tests now assert the expected promoted layout in the JAXPR and compare against dense references instead of self-referential kernel outputs.

Evidence:
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:313)
- [bitpack_binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary_test.py:814)
- [bitpack_binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary_test.py:882)

### B. Binary MV row-gather path was still callable even though the intended implementation had moved

Before:
- The pre-synaptic `transpose=False` binary MV row-gather route still existed as a regular path, which conflicted with the intended move to a column-scatter operator backed by a maintained column-major mirror.
- That made it too easy to accidentally exercise the old route instead of the intended one.

After:
- The route now raises unless a column-major mirror is supplied.
- CUDA dispatch for the intended path now goes through `binary_fcnmv_col_scatter.cu`.
- Non-CUDA fallback and post/scatter misuse are explicit warnings instead of silently pretending to use the mirror.

Evidence:
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:513)
- [binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary_test.py:413)
- [binary_test.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary_test.py:449)

### C. Compact 2D reflected matmul used mismatched metadata

Before:
- In 2D compact reflected matmul, only `other.value` was transposed before calling FCNMM, while `other.packed`, `other.active_ids`, and `other.n_active` still described the untransposed tensor.
- That meant the compact metadata no longer matched the matrix seen by the MM kernel.

After:
- The code rebuilds compact metadata from the transposed value before calling the compact MM path.
- This bug is also documented in the branch’s MM correctness notes.

Evidence:
- [main.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/main.py:1028)
- [main.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/main.py:1686)
- [coba_ei_mm_correctness_notes.md](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_mm_correctness_notes.md:13)

Note:
- Later in this branch family, compact FCN operators were removed entirely, so this fix matters mainly as part of the historical correctness trail and for understanding why those routes were unsafe.

### D. COBA MM route tests previously selected MV-oriented backends and overclaimed route coverage

Before:
- MM route helpers could pick the backend using MV-oriented logic.
- Large-scale COBA batch tests could be interpreted as MM route coverage even when the actual runtime path entered 1D MV dispatch and only then maybe got promoted.
- Some removed routes were still present in benchmark route matrices.

After:
- MM helpers use `preferred_real_mm_backend(...)`.
- MV and MM benchmark tests are split.
- MM benchmark route coverage now excludes removed routes and explicitly skips unregistered `binary/pre/col_scatter` MM.
- Large-scale tracing records more intermediate summaries instead of only spike history.

Evidence:
- [coba_ei_benchmark_test_helpers.py](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_benchmark_test_helpers.py:239)
- [coba_ei_benchmark_mm_test.py](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_benchmark_mm_test.py:35)
- [coba_ei_mm_correctness_notes.md](/home/luruheng/code/brainevent_main_save/brainevent/dev/fcn/coba_ei_mm_correctness_notes.md:69)

### E. `test_colmajor_fullwarp_nocap` needed public-layout repair on the Python side

Before:
- The testing kernel uses a different physical output layout for the transpose case.
- Without Python-side output-shape adjustment and post-call layout restoration, it would not be safe to compare or expose like a normal `binary_fcnmm` backend.

After:
- `binary_fcnmm_p_call()` adjusts output shape when this backend is used on transpose=True.
- `_binary_fcnmm_restore_public_layout()` transposes the raw result back to the public layout.
- MV batching that promotes into MM also applies the same restoration.

Evidence:
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:886)
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:1424)
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:691)

### F. Explicit `backend` used to be soft in practice, and is now effectively strict

Before:
- Some FCN entrypoints behaved as if `backend` was only a hint, because batching promotion and wrapper layers could end up consulting global backend state or default primitive backend state later in the call chain.
- That made it hard to reason about whether a benchmark or test was actually exercising the requested implementation.

After:
- Wrapper/helper code now threads explicit `backend` through to the promoted MM primitive path.
- If an explicit backend is passed and the primitive does not support it on the active platform, dispatch now fails instead of silently choosing some other backend.
- This makes benchmark selection and failure modes much more honest.

Evidence:
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:49)
- [bitpack_binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/bitpack_binary.py:314)
- [binary.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/binary.py:872)
- [main.py](/home/luruheng/code/brainevent_main_save/brainevent/brainevent/_fcn/main.py:139)

## Current caveats / unfinished points

These are important to note so the review does not overstate what is finished.

### 1. `binary_fcnmm_col_scatter.cu` exists but is not yet registered

- The file is present.
- `binary.py` currently does not load or dispatch it.
- The MM benchmark now correctly treats `binary/pre/col_scatter` as unregistered and skips it.

This means MV col-scatter is formally integrated, but MM col-scatter is not yet wired into the Python primitive.

### 2. COBA batch e2e can still mix MV/MM depending on the transformation path

- Direct 2D FCN tests are the clearest proof of MM correctness.
- COBA batch tests may start from 1D MV calls inside `brainstate.nn.Map` and then rely on batching promotion.
- This is particularly important for bitpack post routes, where deleted MV scatter can still surface if the transform chain does not promote early enough.

What this means in practice:

- the batching rule itself was updated correctly,
- but COBA batch execution still depends on how JAX traces and lowers the mapped call chain,
- so the real question is not only "is the batching rule correct?" but also
  "does this version of JAX actually take the promotion path we expect in this benchmark shape?"

### 3. Compact preprocessing improved, but compact FCN operators are intentionally removed

- There is meaningful new JAX fallback support for compact event preprocessing.
- But the FCN compact operators under `_fcn` are now removed stubs.
- So “compact support” is only true at the event-preprocessing level, not as an FCN execution path.

## Version-specific test results

This section separates:

- the user-reported result from the dedicated `0.8.2` environment,
- the directly verified local result from the available `0.8.3` environment,
- and the directly verified local result from the available `0.9.0.1` environment.

### JAX 0.8.2

User-reported result:

- `_fcn` pytest under the dedicated `0.8.2` environment passed completely.
- COBA EI MV and MM route correctness were reported as passing in that environment.
- In particular, the user reported that the MM cases which can mis-enter MV on newer JAX were normal under `0.8.2`.
- Interpreted through the current branch logic, this strongly suggests the
  batching/promotion chain in `0.8.2` was successfully lowering the relevant
  COBA batch MM scenarios into the intended MM primitive path rather than
  getting stranded on the MV-side entry primitive.

This is important because it strongly suggests that at least one part of the MV/MM mixed-dispatch behavior is version-sensitive rather than a pure functional error in the FCN code.

Limit:

- I do not currently have a local `0.8.2` interpreter in this workspace, so I am recording the `0.8.2` result here as a user-observed result rather than a directly re-run local result.

### JAX 0.8.3

Directly verified locally with:

- `/home/luruheng/miniconda3/envs/brainevent-jax9/bin/python`
- `jax==0.8.3`

Confirmed results:

- `dev/fcn/coba_ei_benchmark_mm_test.py`: `21 passed, 4 skipped`
- `dev/fcn/coba_ei_benchmark_mm_test.py --collect-only`: `25 tests collected`

Interpretation:

- The current MM benchmark route set is stable on `0.8.3`.
- The skipped MM cases are intentional route exclusions:
  - unregistered `binary/pre/col_scatter` MM,
  - and the benchmark-level routes we intentionally fenced off because they do not provide a valid explicit MM operator path.
- This is consistent with the idea that on `0.8.x`, the batching/promotion
  behavior is still aligned closely enough with the intended FCN MM dispatch
  model for the curated COBA MM benchmark to behave cleanly.

What this supports:

- The branch behavior on `0.8.x` is consistent with the user’s observation that MV/MM route correctness is fundamentally healthier there than on `0.9.x`.

### JAX 0.9.0.1

Directly verified locally with:

- `/home/luruheng/miniconda3/bin/python`
- `jax==0.9.0.1`

Confirmed MV COBA result:

- `dev/fcn/coba_ei_benchmark_mv_test.py`: `77 passed`

Additional verified MV route-boundary tests added during this review:

- `test_pre_col_scatter_cuda_route_avoids_non_cuda_col_scatter_fallback_warning`: passed
- `test_pre_col_scatter_jax_raw_reference_emits_expected_fallback_warning`: passed

Interpretation for MV on `0.9.0.1`:

- COBA EI MV correctness tests are passing.
- The `binary/pre/col_scatter` story on `0.9.0.1` is:
  - the actual `cuda_raw` route should avoid the non-CUDA fallback warning,
  - the `jax_raw` reference path is expected to emit the fallback warning,
  - and that warning is considered correct/expected, not a route error.

Confirmed MM COBA result:

- `dev/fcn/coba_ei_benchmark_mm_test.py`: `21 passed, 4 skipped`

Interpretation for MM on `0.9.0.1`:

- The curated MM benchmark file passes in the current route set.
- The skipped MM routes are intentional and reflect unsupported or unregistered MM behavior, not random test instability.
- The main version-specific risk on `0.9.0.1` is still batching/promotion:
  broad COBA batch MM scenarios are more likely to remain observable as MV-side
  entry paths before or during lowering, especially in the bitpack post route.

Important historical note for `0.9.0.1`:

- Before narrowing the MM benchmark route set, some COBA EI MM cases did fail because the runtime path entered MV instead of a true MM primitive.
- The key pattern was the bitpack post/batch path:
  - the high-level benchmark intended to exercise MM,
  - but the actual mapped execution could enter `bitpack_binary_fcnmv(transpose=True)`,
- and that MV scatter path has been removed,
- so the failure looked like “MM failed”, while the concrete primitive error actually came from MV dispatch.

This is the version-sensitive symptom that differs from the user’s `0.8.2` observation.

In other words:

- the branch contains a batching-rule fix,
- but `0.9.0.1` still exposes more cases where the benchmark-level call graph
  and the primitive-level call graph do not line up the way we intended,
- so "batching logic fixed" and "all COBA batch MM executions reliably appear
  as MM in the final lowered program" are not identical statements on `0.9.x`.

### JAX 0.9.0.1 environment instability notes

There is also a separate environment/runtime issue on the `0.9.0.1` GPU stack that should not be confused with route correctness:

- repeated or broader reruns of `col_scatter`-heavy subsets could trigger:
  - `cudaErrorMemoryAllocation`
  - XLA autotuning failures like `NOT_FOUND: No valid config found!`
- once the runtime gets into this state, even unrelated primitive setup such as `PRNGKey`, `jnp.arange`, or connection preprocessing may start failing.

Interpretation:

- This looks like a JAX/XLA/GPU runtime instability issue under repeated compilation/execution pressure.
- It is not evidence that the FCN route logic itself is wrong.
- For this reason, the most trustworthy `0.9.0.1` conclusions are the clean single-run or targeted-run results listed above, not every failure seen after the runtime becomes unstable.

### Practical conclusion across versions

- `0.8.2`:
  user-reported full `_fcn` pytest pass, and COBA EI MV/MM route correctness looks clean.
- `0.8.3`:
  local MM COBA benchmark also looks clean in the curated route set.
- `0.9.0.1`:
  MV COBA tests pass, and the curated MM COBA benchmark passes, but this version is more prone to:
  - MV/MM mixed-dispatch symptoms in broad COBA MM scenarios,
  - and separate GPU/XLA autotuning instability under repeated heavy runs.

### Batching-specific conclusion

The batching story across versions is:

- On `0.8.2`, based on the user’s result, the practical COBA MV/MM batching
  behavior looks correct end-to-end.
- On `0.8.3`, the available local evidence is still consistent with that.
- On `0.9.0.1`, the branch’s batching-rule changes are real and necessary, but
  they are not sufficient to guarantee that every benchmark-level COBA batch MM
  scenario will present itself as MM all the way through the observed runtime
  path. Some scenarios still expose MV-side entry behavior first, and that is
  exactly where the historical bitpack post/MM-into-MV symptom came from.

## Short summary

The branch does contain the four changes you listed, but the real scope is larger:

- bitpack MM axis/backend propagation was fixed end-to-end,
- MV col-scatter was formally reintroduced on a new dedicated path,
- several old FCN routes were intentionally removed,
- batching promotion logic was rewritten,
- a dedicated `binary_fcnmm` testing backend was added and tested,
- compact preprocessing gained JAX fallbacks,
- benchmark/test structure was reorganized around explicit route semantics,
- and several correctness bugs were fixed or at least surfaced and fenced off.

The most important correctness-oriented fixes are:

- bitpack MV→MM promotion now preserves pack-axis/backend intent,
- binary MV no longer silently uses the wrong pre row-gather path,
- compact 2D reflected MM no longer mismatches transposed values with stale compact metadata,
- and `test_colmajor_fullwarp_nocap` now has explicit public-layout repair so it can behave like a real `binary_fcnmm` backend in tests.
