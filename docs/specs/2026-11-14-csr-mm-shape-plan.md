# Shape-only plan for the CSR hybrid scatter MM

## Goal

The native hybrid MM (`W.T @ events`, kernel `batched_spmm_v3`) runs only when
the matrix was built eagerly on a GPU. Construction then stores host row
statistics, a long-row mask and GPU properties in `CSRConfig`.

brainspike builds `CSR` inside `jit` from traced arrays, so its batched
products always take the JAX scatter. The same happens to a matrix built on the
CPU and moved to the GPU.

The kernel does not need that state:
- any long-row mask or lane group gives correct results;
- an undersized task queue only slows it down.

Plan the call from static shapes and the lowering-time device instead. Then
every construction path lowers to the native kernel, and redrawn structures
share one executable. The prepared-plan machinery can then be deleted.

## Evidence

Baseline: `examples/csr_hybrid_mm_plan_benchmark.py run` on one RTX 3090
(6 MiB L2), jax 0.12.0.1, brainevent at `8bb7731`.
- The full run takes 75 min; `--quick` takes 2.5 min.
- Tables come from `report`. The raw JSON lines are not committed.

Variants:
- **P:** today's prepared path.
- **J:** `jax_raw`.
- **T:** a CSR built inside `jit`, as brainspike does.
- **S:** a prototype that calls the same kernel with a shape-only plan and a
  per-call JAX long-row mask.
- **S0:** S with the mask precomputed.

Every S result matched P.

| gate | condition | measured | pass |
|---|---|---|---|
| G1 | S <= 1.10x P (geomean, batch >= 8); every case S - 5 us <= 1.25x P | geomean 0.998, worst 1.08 | yes |
| G2 | S >= 1.5x faster than J (geomean; M, L; batch >= 8; activity <= 20%) | geomean J/S 6.97 | yes |
| G3 | mask <= 5% of an S call (median, batch >= 8) | median 0.7%, max 27.2% | yes |
| G4 | S scratch <= max(2x P, 25% of the matrix storage) in every case | max S / limit 1.12 | **no** |
| G5 | S compiles once over redrawn structures and lowers natively | S [1, 1, 1], P [1, 2, 1] executables; S native | yes |
| G6 | shape-based group within 5% of the best group in >= 90% of sweeps | 92% within 5%, worst 1.14x | yes |
| G7 | 'worst' capacity <= 1.10x 'exact' on hub and Pareto rows | worst/exact max 1.01 | yes |
| G8 | one constant chunk within 5% of the L2 rule in every sweep | best constant 4: 1.50x of L2 rule at worst | **no** |
| G9 | bounded SDDMM capacity cuts its scratch >= 4x | 1.3x to 1.5x | **no** |

Time ratios use geomeans over six row-length distributions, binary and
weighted events, and batch 8/64/256.

| size (rows x cols, mean row) | activity | S/P | J/S | T/J |
|---|---|---|---|---|
| S (4k x 4k, 80) | 1% / 5% / 20% / 100% | 1.02 / 0.99 / 0.99 / 0.99 | 6.1 / 4.0 / 1.9 / 0.55 | 1.00 |
| M (20k x 20k, 200) | 1% / 5% / 20% / 100% | 1.02 / 1.01 / 1.00 / 1.00 | 17.0 / 7.6 / 2.6 / 0.62 | 1.00 |
| L (100k x 100k, 100) | 1% / 5% / 20% / 100% | 1.01 / 0.99 / 0.98 / 1.00 | 14.9 / 7.9 / 2.9 / 0.68 | 1.00 |
| W (10k x 200k, 1000) | 1% / 5% / 20% / 100% | 0.99 / 0.99 / 1.00 / 1.00 | 5.1 / 3.3 / 1.8 / 0.59 | 1.00 |

Other measurements:
- **Lowering.** The native kernel is reached by an eager GPU build and by vmap
  of MV over it. It is not reached by a build inside `jit` or a CPU build.
  T/J = 1.00 everywhere: the brainspike path is the JAX scatter, so S would
  make it 2–17x faster at ≤ 20% activity.
- **Mask.** The long-row mask costs 2.2–7.3 µs per call (4k to 100k rows). In a
  loop over fixed connectivity XLA computes it once; S0 ≈ S.
- **Chunk.** The L2 rule is within 1.09x of the best constant on M and L.
  - Constant 16 is best on M but 2.8–4.4x slower on L and W.
  - Constant 4 is best on L and W but 1.5x slower on M.
  - On W at 5% activity, the L2 rule picks chunk 1 and runs 1.5–1.7x slower than 4.
    P does the same today.
- **Capacity.**
  - The 'worst' bound is within 1.01x of exact.
  - `ceil(nnz/512)` alone overflows on W with hub rows: 1.42x slower.
  - A zero-capacity queue is 1.5–4.4x slower.
- **Group.** The shape-based group is within 5% of the best in 92% of sweeps.
  The worst is 1.14x, on W with Pareto rows at 5% activity.
- **Scratch.** The largest per-size scratch is:

  | size | P | S | matrix storage |
  |---|---|---|---|
  | S | 2.3 MiB | 3.4 MiB | 2.4 MiB |
  | M | 2.6 MiB | 6.0 MiB | 30.5 MiB |
  | L | 1.5 MiB | 2.9 MiB | 76.3 MiB |
  | W | 0.5 MiB | 0.5 MiB | 76.3 MiB |

  G4 fails only on S with Poisson rows, by at most 1.9 MiB. No row there is
  long, so P reserves no queue.
- **Other dtypes.** f16, bf16, f64 and shared weights match P within 1%.
- **SDDMM.** The matrix SDDMM gradient's worst-case scratch is 53 MiB at
  100k x 100k. A bounded queue cuts it only to 39 MiB.
- **Outside this spec.** With every event active, J is 1.5–1.8x faster than the
  native kernel.

## Decisions

- **D1 Plan inputs.** The MM plan is a pure function of these inputs:
  - rows, cols, nnz and batch;
  - value dtype and `weighted`;
  - the lowering target's L2 size.

  `CSRConfig` stops carrying MM state. *Why:* G1, G2 and G5. The plan never
  reads arrays, so traced, CPU-built, jit-built and vmapped structures all lower
  natively and compile once.
- **D2 Long rows.** The CUDA kernel callable computes the mask during lowering
  with JAX ops: `diff(indptr) > 16·G`, packed into uint32 words. The SDDMM
  `_long_mask` becomes the same helper with its own threshold. *Why:* G3. The
  mask costs a few µs, and loops over fixed connectivity compute it once. A
  native row-classification kernel is not needed.
- **D3 Group.** `quarter = (min(nnz // rows, 512) + 3) // 4`, rounded up to 4,
  8, 16 or 32. *Why:* G6. No per-matrix group hint is added unless a workload
  shows a need.
- **D4 Capacity.** `long_tasks = ceil(nnz / 512) + min(rows, nnz // (16·G + 1))`
  per batch chunk. This is a true upper bound, so the queue cannot overflow.
  *Why:* G7, and the 1.42x overflow cost of the smaller bound.
  - **For approval:** G4 fails as written, by at most 1.9 MiB, on a 2.4 MiB
    matrix whose rows are all short. I propose accepting it and amending G4 to
    allow `max(2x P, 25% of matrix storage, 4 MiB)`.
- **D5 Chunk.** Keep today's rule:
  `chunk = min(batch, 65535, (L2 / 4) // (cols · max(4, value bytes)))`.
  L2 is read at lowering from the target devices, taking the minimum over the
  candidates. *Why:* G8 fails, because no constant chunk is near-best on every
  shape, so the device matters.
  - Move `_tcsr/_resources._target_devices` into `_op/`.
  - `XLACustomKernel` passes the target devices to kernel generators that
    declare a `target_devices` keyword.
  - L2 comes from `cuda_driver.device_attributes`, cached per device. With no
    usable device, use the smallest local CUDA L2, or 4 MiB.
  - Drop:
    - the `expected_sms` / `expected_l2_bytes` attributes;
    - the prepared-device check;
    - the `_support/device_info.cu` fallback.
  - The W chunk-1 gap is unchanged from today; it is a separate tuning question.
- **D6 API.** Remove the unreleased MM-planning surface:
  - the types `MMStructureSpec`, `MMStructurePlan`, `MMCallPlan`,
    `CSRDeviceInfo` and `get_device_info`;
  - the structure and device fields in `_ConfigAux` and `CSRKernelSpec`;
  - `CSRConfig.prepare_structure` and `CSRConfig.mm_plan`;
  - `StructureCache.mirror_preparation` and the mirror-plan sync helpers;
  - the `device=` parameters, and `with_rebuilt_config` on CSR, CSC and FCN;
  - the plan-free fallback `PerformanceWarning`, and the forced `jax_raw` for
    transposed matrix events.

  Keep `CSRConfig` (MV schedule and tuning records), `MVSchedule` and
  `load/save/init_csr_config`. The tuning store's device kind comes from the
  structure's device. *Why:* none of this is in a release (0.4.2 still has
  `HybridConfig`), and brainspike uses none of it.
- **D7 Operands.** The MV, MM and indexed primitives drop the `long_mask`
  operand. Their JVP and transpose rules derive MM plans from shapes.
- **D8 SDDMM.** No kernel change; the worst-case sizing stays. *Why:* G9. A
  bounded queue would only save 1.3–1.5x and would need overflow handling in two
  kernels.
- **D9 Backends.** Add a public, warning-free
  `XLACustomKernel.resolve_backend(platform, requested=None)` and use it at all
  12 call-time resolution sites.
  - Change the indexed-route default for `W @ x` and the compact-binary
    preprocessing.
  - Keep the FCN gather backend and TCSR tile building.
  - Fix the SDDMM bug, test first: a global `cuda_raw` with `transpose=False`
    sends `'auto'` leaf calls to the CUDA kernel, which raises.
- **D10** Update the docs and the changelog.

## Step plan

1. **Failing tests.**
   - Built inside `jit`, built on the CPU, vmapped and eager structures all
     lower to `batched_spmm_v3` (checked via the HLO custom-call target).
   - Ten redrawn structures share one executable.
   - The SDDMM `'auto'` bug.
2. **Lowering-time target devices** in `_op/`, with the cached L2 query (D5).
3. **Static MM plan** (D1–D4). The kernel callable builds the mask; the CUDA
   source loses only the device-check attributes.
4. **Drop the `long_mask` operands** (D7).
5. **Slim the API** (D6): `CSRConfig`, `StructureCache`, the constructors, FCN
   and TCSR.
6. **Backend resolution** (D9).
7. **Docs and changelog** (D10). Then re-run the benchmark and compare it with
   the baseline.

## Verification

- **Benchmark:** `report base.jsonl new.jsonl`.
  - The new P is within 1.05x of the baseline P (geomean, batch >= 8).
  - T now equals the native path: T/J ≈ 1/J-over-S above.
  - S/P stays within G1.
- **Tests:**
  - full CPU suite;
  - GPU suites for `_csr`, `_tcsr`, `_sddmm`, `_fcn` and `_op`;
  - mypy on the touched modules;
  - brainspike's tests against this branch.

## Risks

- **No device assignment at lowering** (AOT export, some `shard_map` cases).
  The candidate-device or default L2 affects speed, not correctness.
- **Single GPU model.** Only the RTX 3090 (6 MiB L2) was measured. The L2 rule
  was originally tuned on a larger-L2 GPU.
- **Mirror memory.** Route changes under D9 may build the mirror lazily, which
  costs 2·nnz extra index words.
