# Integrate prepared delay CSR into the main package

## Goal

Replace `brainevent/_csr/delay` with the existing implementation from
`deving/operators/delay_mv`, excluding cuSPARSE. The installed package must
import and execute without the development directory.

## Decisions

- `delaycsr(CSR)` is the only delay layout object. `CSR.delay_layout(delays,
  n_slots)` constructs it from concrete per-edge delays outside JIT and
  inherits the source CSR backend. Remove the old `CSRDelayLayout` class,
  `CSR.delay_deliver`, and non-layout `csr_delay_deliver` function/primitive.
- Keep `csr_delay_deliver_layout(ring, events, layout, t, *, weighted=None,
  backend=None)` as the pure, additive API. It accepts an initialized
  `delaycsr`, returns the full ring, and leaves the object's state unchanged.
- Preserve the development implementation's stateful `forward`, `x @ W`,
  JAX `clear_head`, prepared workspace, pytree, JVP/VJP and batching rules.
  Carry over the overflow-safe slot indexing.
- `with_data` accepts a weight array in the initialized object's current
  edge order, with unchanged shape, dtype and unit. It never reapplies the
  original CSR permutation. Initialization remains outside JIT/AD.
- Explicit backends are `cuda_raw` and `jax_raw`; `None` selects the platform
  default through the inherited CSR backend. Do not restore `cuda`, cuSPARSE,
  or the previous non-layout/Numba implementations.
- Keep the validated development dtype/input contract. This migration does
  not add Quantity support, mixed weighted dtypes or scalar-delay broadcasting.
- Use a relative CSR import to avoid partially initialized top-level imports.
  Put CUDA trait headers under `delay/_common`, resolve loader paths relative
  to the installed package, and include all CUDA sources/headers in wheels.
- Preserve manual sharding axes on runtime-check callback results using the
  main library's compatibility helper. Keep the check's custom JVP as the
  identity so reverse mode does not introduce a cross-shard cotangent sum.

## Steps

1. Transfer the production Python and hybrid CUDA code; remove obsolete files.
2. Wire package exports and `CSR.delay_layout`; update the live API reference
   and benchmark to the prepared-layout interface.
3. Migrate development tests and update cross-operator transformation tests.
4. Verify source-tree and installed-wheel behavior on CPU and GPU.

## Verification

- Import from the top level and subpackage without circular imports; check
  exports and the absence of obsolete non-layout entry points.
- Exercise binary/weighted events, shared/per-edge weights, empty graphs,
  repeated destinations, time wrapping, backend inheritance and rejection.
- Check pure delivery, stateful clearing, JIT/scan, vmap, JVP/VJP and plasticity
  updates using current-order `with_data`.
- Keep int32-overflow regressions without allocating a large ring. Leave the
  existing multi-billion-edge stress test opt-in.
- Build a wheel, inspect its assets and execute CPU/GPU delivery and gradients
  with only the unpacked wheel on `PYTHONPATH` from outside the checkout.

## Results

Validated with Python 3.12 and JAX 0.11.0:

- Final delay suite: CPU 318 passed, 301 skipped, 3 deselected; GPU 618
  passed, 1 skipped, 3 deselected. Large-allocation cases remain opt-in.
- Existing operator sharding regressions: 13 passed on two CPU devices and
  17 passed on GPU. Runtime checks still reject invalid per-shard heads and
  segment counts, and a subsequent valid call succeeds.
- GPU production Python line coverage: 94.25%, excluding test helpers.
- Whole-package mypy with `--no-site-packages`: 178 source files passed.
- Wheel built successfully with all five CUDA source/header assets. From an
  external directory using only the unpacked wheel, CPU and GPU passed pure
  delivery, gradients, sharded loops and JAX clearing with every available
  delay backend (including the automatic default).
- Ordinary CSR CPU regression: 84 passed, 4 skipped, 468 deselected; two
  additional cases required the environment's missing optional Numba backend.
  Both passed when rerun with the CPU default temporarily set to `jax_raw`.
