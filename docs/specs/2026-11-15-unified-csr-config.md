# Unified CSR scheduling configuration

## Goal

Keep one immutable `CSRConfig` for CSR MV and hybrid MM scheduling, removing
`MVSchedule`, `_MMPlan`, `_BenchmarkRecord`, and `_ConfigAux`. Preserve the
default kernels, scheduling formulas, automatic configuration selection and
mask lifetime. Callers still need no preparation step.

The preceding mask-cache implementation is committed and pushed as `8f1403d`
(`perf(csr): reuse structural MM masks across calls`).

## Decisions

- Flatten the four existing MV fields into `CSRConfig`, preserving their
  names and positional order. Add keyword-only MM policy fields:
  `mm_l2_divisor=4`, `mm_sparse_l2_divisor=2`,
  `mm_sparse_l2_min_bytes=32 << 20`, `mm_short_rows_floor=1 << 20`, and
  `mm_max_chunk=65535`.
- Keep MM iterations (16) and task edges (512) as fixed class constants,
  matching the CUDA implementation. This change does not expose unsupported
  kernel geometry overrides.
- Put MV compiler flags, module suffix and queue bounds, plus MM group,
  threshold and plan calculations, on `CSRConfig`. `mm_plan` returns a fresh
  dictionary of the seven current plan attributes. Any internal plan cache
  stores immutable tuples, never shared mutable dictionaries.
- Keep resolved plans local to lowering: shape, batch after vmap, dtype,
  event mode and target L2 remain call inputs, not persistent configuration
  fields. Preserve the current formulas exactly under default settings.
- Use dictionary benchmark records, defensively copying supplied mappings.
  Exclude diagnostics from equality and hashing. Use the immutable config
  itself as pytree auxiliary data, with zero array leaves.
- Preserve the primitive parameter name `config_spec`, replacing its value
  with `CSRConfig`. Forward all fields through batching, differentiation and
  CSR fallback. TCSR's separate tile algorithm is unchanged.
- Keep masks on `StructureCache`, as dynamic leaves of the matrix. Mask
  construction and MM lowering share one threshold method; policy fields
  introduced here do not change row classification.
- Centralize flat parameter serialization on `CSRConfig`. Continue reading
  existing four-field JSON configuration and environment overrides, defaulting
  missing MM fields. Persist policy fields, not resolved plans or masks.
  Existing initialization continues to benchmark MV candidates only.

## Structure

```text
CSR / CSC
├── config: CSRConfig (frozen, zero array leaves)
│   ├── four flat MV fields
│   ├── five keyword-only MM policy fields
│   ├── fixed MM kernel constants
│   ├── diagnostic records: tuple[dict]
│   ├── validation and flat parameter serialization
│   ├── MV flags, module suffix and task capacity
│   ├── shared MM group and long-row threshold
│   ├── mm_plan(...) -> fresh temporary dict
│   └── pytree flatten/unflatten (aux = config)
└── _cache: StructureCache
    ├── primary_mm_mask
    ├── mirror_mm_mask
    ├── mirror
    ├── diag_plan
    └── tile_geometries
```

The class lives in `_csr/config.py`. Forward consumers in `_csr/binary`,
gradient consumers in `_sddmm`, and the TCSR CSR-fallback path pass the same
config. `_csr/binary/initialize.py` retains benchmark and store ownership.
`_structure.py` owns the cached arrays. FixedNum's indexed mirror path also
uses this configuration and its corresponding cached mask.

## Interface changes and downstream boundaries

`CSRConfig(block_size=...)`, matrix constructors, public raw MV/MM signatures,
`long_mask`, and configuration load/save/init entry points retain their calling
conventions. Remove public `MVSchedule` exports and `CSRConfig(mv=...)`;
`config.mv.field` becomes `config.field`. Record attribute access becomes
dictionary indexing. Private plan and auxiliary types disappear.

The inspected local brainspike and braintrace call sites do not use the
removed types. Preserve primitive names, `config_spec`, and CSR's outer pytree
layout for their existing calls. Old pickled auxiliary objects or checkpoint
states containing a persisted CSR treedef are not guaranteed to load across
this API change. Perf-Benchmark's already removed `prepare_structure` and
`get_device_info` dependencies are outside this refactor. Brainspike's existing
per-timestep CSR reconstruction still rebuilds masks.

## Implementation and verification

1. Implement the unified class, shared mask threshold and immutable plan cache.
2. Migrate native forward/gradient/fallback consumers, tuning persistence,
   exports, documentation, and active development plan-verification scripts.
   Keep frozen benchmark source snapshots unchanged.
3. Migrate and extend existing pytest modules only. Check default plan
   equivalence, non-default MM policy propagation, old-store loading, fresh
   plan dictionaries, diagnostic-insensitive JIT identity, and scan/vmap/AD.
4. Run focused CPU and GPU regression suites, mypy and diff checks. Recheck
   brainspike storage/checkpoint and braintrace paths using the modified tree.
   Reproduce failures against the pushed baseline before attributing them to
   this refactor.

## Implementation verification (2026-11-15)

The unified implementation is present in the worktree. No new pytest files
were added. Compared with the pushed `8f1403d` source, 10,000 generated call
metadata combinations produce identical default MM plan dictionaries. The
four public raw MV/MM function signatures are unchanged.

Completed checks:

| Check | Result |
| --- | --- |
| Config and MM plan CPU tests | 140 passed |
| Core config statement coverage | 98.86% (173/175) |
| Complete ordinary/indexed, SDDMV and TCSR consumer CPU tests | 1,985 passed, 554 skipped |
| Configuration initialization and persistence, CPU | 63 passed, 1 GPU-only skipped |
| Configuration initialization GPU smoke | 1 passed |
| CSR/FCN/TCSR lifecycle and SDDMM dispatch, CPU | 562 passed, 76 skipped |
| Native consumers, config propagation and AD, GPU | 113 passed |
| Additional native MM and supplied-mask GPU cases | 94 passed |
| Shape-plan/scan, event gradients, structure and automatic tiles, GPU | 292 passed |
| Brainspike storage and legacy projections, CPU | 107 passed |
| Brainspike device-tuned checkpoint checks, CPU | 2 passed |
| Braintrace sparse operations, CPU | 37 passed |
| Mypy without runtime dependencies | 174 source files clean |

Brainspike `branch:grouped`, `k1a:coba` and `k1b:weighted_count` have identical
placement decisions and program identity digests under the pushed baseline
and the unified config. The local machine-readable comparison is saved at
`dev/md/2026-11-15-unified-csr-config-verification.json` in the surrounding
workspace; the Markdown link there points to this spec.

The extra GPU selection also matched four explicitly CPU-only tests. They
fail identically on the baseline: two request the unsupported Numba GPU
platform and two compare against a dense TF32 GPU reference with a CPU-level
tolerance. They are not new native-MM failures. No CUDA kernel source changed.
