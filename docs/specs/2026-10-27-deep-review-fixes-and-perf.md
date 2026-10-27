# Deep review: correctness fixes, dynamic-structure pytrees, exact JITC density (0.4.0)

Branch: `fix/deep-review-0.4.0`. Three stacked PRs, one per phase.

## Goal

A whole-package review (three parallel code reviews plus dense-reference probes over every
matrix class, operator, transform and backend on 2x RTX 3090) found that the kernels agree with
the dense reference everywhere, but the layers around them have crashes and silently wrong
results. This change fixes them and the performance problems found on the way.

The worst ones:

- A CSR/CSC/FixedNum matrix closed over by two jitted functions raises `UnexpectedTracerError`
  after its first binary matmul (the lazily built task workspace is a tracer cached on the
  object). A second, structurally equal matrix passed to the same jitted function raises
  (index arrays in pytree aux data). A matrix carried through `lax.scan` with a binary matmul
  raises (the lazy cache changes the pytree structure).
- Silently wrong output: `events @ csr.T` on `cuda_raw` after a prior `events @ csr` (stale
  workspace capacity); `BinaryArray @ FixedNum` when the batch equals `n_post`
  (shape-equality layout heuristic); `CSC.solve`; JITC normal weight gradients on GPU under
  x64; functional `csrmm` with a non-float32 `B` on GPU; JITC density `2/ceil(2/p)` instead of
  `p` (0.667 for every `p` in `[2/3, 1)`).
- Performance: index arrays are embedded as HLO constants (160 MB HLO and 10-30x slower
  compile for 10M nnz); `set_numba_parallel` is ignored once any kernel is registered; JITC
  `events @ M` takes the non-event-driven gather path by default (280 vs 58 µs).

## Decisions

1. **Structure becomes dynamic pytree leaves for CSR, CSC and FixedNum** (TCSR already works
   this way). It removes the aux-equality crash and the HLO-constant compile cost by
   construction, and lets binary products run with traced structure. The cost is a breaking
   change: `jax.grad(f)(M)` needs `allow_int=True`; differentiate `M.data` and rebuild with
   `M.with_data(d)` instead (TCSR already requires this). `tree_unflatten` builds objects
   without validation so `allow_int=True` and filter-style transforms keep working.
2. **The hybrid task workspace never lives on the matrix.** The hybrid kernels zero `status`
   and write `task_begin/task_end` before reading them, so they are pure scratch: capacity is
   bounded from static shapes (plus an optional, config-independent `max_row_nnz` hint) and
   the scratch is created inside the kernel lowering. `workspace=` on the public binary
   functions becomes optional and deprecated. No lazy cache mutates a matrix under a trace.
3. **Mirrors (CSC view of a CSR and vice versa) use orientation-free names** ("primary" and
   "mirror") in a cache shared by `with_data`, `apply`, binary ops and `.T`, so transposition
   never re-keys anything. Built lazily only when the structure is concrete and no trace is
   active; with traced structure and no mirror, operations fall back to mirror-free kernels.
4. **JITC density is exact by thinning.** The walk uses `cl = floor(2/p)` (or `round(2/p)`
   when `2/p` is an integer within tolerance) and keeps each candidate edge with probability
   `p*cl/2`, decided by a stateless 24-bit hash of `(seed, stream key, position)` compared
   against an integer threshold. Integer comparison keeps numba, pallas and CUDA bit-identical;
   keyed by the same pair as the per-edge weight hash, so forward, transposed and gradient
   operators regenerate the same matrix. When `2/p` is an integer the threshold means keep-all
   and connectivity is bit-identical to 0.3.0. `prob` is validated.
5. **JITC classes default to `corder=True`**, which keys the RNG streams by rows so
   `events @ M` is the event-driven scatter and `M @ v` a per-row gather. Verified before the
   change: with `corder=True` every product, transpose, grad, vmap and jit agrees with
   `todense()` on numba, jax_raw and cuda_raw. Breaking: pass `corder=False` to reproduce 0.3.0
   matrices.
6. **Numba parallel stays off by default.** `set_numba_parallel` is fixed: the kernel
   fingerprint includes numba's target options, and changing the setting clears JAX caches.

## Plan

1. **Phase 1 (PR 1): localized fixes**, each with a failing co-located test first:
   CSR/CSC binary ops (`CSR * CSR`, `± scalar`), constructor `data` size check, `CSC.solve`
   and `CSR.solve(tol, reorder)`, FixedNum matmat layout (static flag instead of shape
   heuristics, incl. JVP/VJP) and dense `*`/`/`, JITC normal/uniform scalar and unary ops
   and `.sum()`, JITC gradient dtypes, dtype promotion in functional `csrm*`/`jit*m*`,
   `CompactBinary` dispatch, TCSR gradient/nnz=0/batching fallbacks, delay-deliver f16/bf16
   events on numba, `slice_rows` CUDA accumulation, symbolic-zero cotangents, JITC benchmark
   data, lazy compile-cache directory, degenerate LFSR seeds, `set_backend` docs and platform
   validation, numba fingerprint.
2. **Phase 2 (PR 2): pytree and workspace redesign** (decisions 1-3), then the `.cu` change
   that makes scratch FFI outputs instead of inputs.
3. **Phase 3 (PR 3): JITC generation** (decisions 4-5): shared CUDA walk header, keep-hash in
   numba/pallas/CUDA, applied to every walk site (float, binary, `*_to_csr`, `dt2t`, three
   families); class default `corder=True`; docstrings.
4. Changelog `[0.4.0]` with migration notes.

## Verification

- Per phase: new tests fail before and pass after; `pytest brainevent -m ""` on CPU and GPU;
  `mypy brainevent`.
- Dense-reference probe over every class x operator x transform x backend (x32 and x64), plus
  edge cases (nnz = 0, empty rows, duplicates, 1xN): zero failures.
- Regression tests per bug, including: two jits over one closed-over matrix plus
  `jax.clear_caches()`; `f(M1, x); f(M2, x)` hits one cache entry; scan/fori carry; heavy-row
  capacity checks in both orientations and after `.T.T`; `jax.eval_shape` and `jit(f).lower`;
  HLO size independent of nnz; JITC density within 4 sigma of `p` over a grid; bit-identity with
  0.3.0 golden matrices for integer `2/p`.
- Benchmarks (in PR descriptions): JITC `events @ M` on GPU (~280 -> ~60 µs at n = 100k, 1 %);
  10M-nnz CSR compile time and HLO size; `set_numba_parallel(True)` after first use takes
  effect.

## Implementation notes (Phase 2)

- Decision 3: the mirror is also cached when the structure is concrete under an active trace
  (a closed-over matrix). It is computed eagerly, so no tracer is stored; only a traced
  structure takes the mirror-free path.
- Decision 2: the `max_row_nnz` hint lives in the pytree metadata, rounded up to a power of
  two so re-drawn matrices of one family keep one compilation. It keeps `task_capacity` at 0
  for matrices without heavy rows, which skips the second hybrid kernel launch (the static
  bound alone would add it to every call). The public functions reject an understated hint
  when `indptr` is concrete.
- The `.cu` change (scratch as FFI outputs) was dropped: per-step times with in-lowering
  scratch match 0.3.0 within 1 µs on an RTX 3090 (`lax.scan`, 4k-100k neurons).

## Implementation notes (Phase 3)

- `(cl, thr)` is computed with the same float32 `jax.numpy` ops whether `prob` is concrete or
  traced (a JITC matrix passed into `jit` traces it), so both give the same walk, also under
  x64. The functional APIs are jitted, so the concrete check runs in a new `namescope`
  `precheck` hook, before the jit boundary.
- The walk stride is capped at `2**25` (keeps `lane + 32 * q` in `uint32`); sparser `p`
  thins from the cap.
- The CUDA keep test lives in `brainevent/include/jitc_walk.h` (a `.h`, so the build cache
  hashes it). The per-file copies of the other walk helpers were not merged: the change
  only adds the keep test at each walk site.
- Cost of the keep hash, when `2 / p` is not an integer: none on scatter products; +28%
  (scalar) and +11% (normal) on gather products (RTX 3090, n = 100k). The default flip makes
  `events @ M` 438 -> 55 µs and dense `x @ M` 428 -> 1775 µs; both are in the changelog.
