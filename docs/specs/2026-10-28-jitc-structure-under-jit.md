# JITC `tocsr` / `dt2t` under `jit`, `vmap` and `grad`

## Goal

`jit*_to_csr`, `M.tocsr()` and the JITC `dt2t` methods failed inside any
`jax.jit` or `jax.vmap`, even with the matrix closed over: the count pass
was bound inside the trace, so `device_get(row_counts)` met a tracer. The
`corder=False` path also sorted indices on the host. Make them work under
`jit`, `vmap` and `grad`, make `tocsr().data` differentiable with respect to
the weight parameters, and support matrices passed into `jit` as arguments
when they carry a prepared structure.

## Decisions

1. **Structure at trace time, weights in the graph.** The connectivity
   depends only on `(shape, prob, seed, corder)`. It is drawn under
   `jax.ensure_compile_time_eval()` with a placeholder weight. The weight
   fill stays in the graph, so the weights may be traced. The output shape
   is `(nnz,)`, and it is static.
2. **Deterministic fill, fixed permutation (user choice B).** The
   `corder=False` fill scatters with atomics, so its order within a row
   changes between GPU runs and a precomputed permutation would not apply.
   Both `corder` values therefore fill with the deterministic row-stream
   (`corder=True`) kernel:
   - `corder=True` fills on `shape` directly;
   - `corder=False` fills on the transposed shape, which draws the same
     matrix bit for bit (verified on CPU and GPU, 180 cases each).

   A fixed `order` maps the fill sequence to the canonical column-sorted
   CSR.
3. **Structure as data.** The structure is stored as
   `JITCStructure(fill_offsets, indptr, indices, order)`. By default it is
   computed at trace time and embedded as constants, so the HLO grows with
   nnz (accepted).
   - `M.prepare_structure()` stores it on the matrix as pytree leaves. A
     prepared matrix passed into `jit` then carries the structure as
     arguments, so it is not embedded. nnz comes from the leaf shapes, so a
     traced `prob` / `seed` is also accepted.
   - Operations that change `prob`, `seed`, `shape` or `corder`, including
     `.T`, drop the structure. Weight-only operations keep it.
4. **Traced `prob` / `seed` without a prepared structure raise** a
   `ValueError`. The message names `prepare_structure()` and closing over
   the matrix.
5. **Gradients in `to_csr`.** The `_with_linear_param_grads` custom JVP
   moves from `dt2t` into `to_csr`, so both `tocsr().data` and `dt2t` are
   differentiable. Scalar weights need no fill at all:
   `data = full(nnz, weight)`.

## Steps

1. Add `brainevent/_jitc_structure.py`. It holds `JITCStructure`, the
   structure builder (trace-time, concrete, raises on tracers) and the
   runtime assembly, all family-agnostic.
2. Rewrite `jit{s,n,u}_to_csr` on top of it, adding `structure=None`.
   Rewrite `jit{s,n,u}mv_dt2t` on top of `to_csr`, also adding
   `structure=None`.
3. Change the JITC classes:
   - add `prepare_structure()`;
   - store `_structure` in the pytree children, with a flag in `aux`;
   - propagate it through `with_data` and `_new_mat`, and drop it in
     `transpose` and when `prob` / `seed` change;
   - pass `structure=` from `tocsr` / `dt2t`.
4. Update the docstrings that call `tocsr` eager-only, and add a changelog
   entry.

## Verify

- Three families × both `corder` values × thinned and integer `2/p`. Each
  of the following must be bit-identical to eager:
  - `jit` with the matrix closed over;
  - `jit(grad)` and `grad(jit)`;
  - `vmap` over `y` and over the weights;
  - a prepared matrix passed as a `jit` argument, including a traced seed.
- `tocsr()` under `jit`.
- `check_grads` (reverse mode) for `tocsr().data`.
- An HLO size check: a prepared matrix passed as an argument stays small;
  closing over it embeds the structure.
- The error message for traced `prob` / `seed` without a structure.
- Structure dropping in `.T` and when the seed changes.
- Full CPU and GPU suites; mypy.
