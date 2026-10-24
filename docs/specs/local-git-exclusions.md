# Local Git exclusions for development and generated data

## Approved scope

Apply local Git exclusions to the ten repositories currently located under
`/home/luruheng/BrainEvent`, including nested repositories. Exclude development
directories and connectome caches/generated data while retaining normal source
tracking. Keep all working files on disk. Do not commit, push, rewrite history,
delete existing object packs, or change global Git configuration.

## Implementation

- Store the rules in each repository's Git common directory under `info/exclude`.
  Preserve existing rules and use an identifiable, idempotent managed block.
- Match `dev` directories at any depth and inspect explicitly named development
  directory variants before including them. Exclude connectome cache directories
  and identified generated simulation/profiling artifacts. Do not broadly ignore
  every NumPy file, source directory, or the entire connectome project.
- Back up each affected repository's index, exclusion file, branch identity,
  indexed paths, and staged state before changing settings or the index.
- Remove matching, already-indexed files from the index only. Preserve all
  unrelated index entries and pre-existing working changes. Perform index changes
  only on existing non-main/non-master worktree branches.
- Record the complete rules, affected repositories, removed index entries,
  validation results, and backup location after execution.

## Reproduction and acceptance checks

Before modification, demonstrate that the large connectome cache outside `dev`
is not ignored and that some `dev` files remain indexed despite ignore rules.
After modification:

1. Check root-level and deeply nested `dev` paths, paths containing spaces and
   Chinese characters, and real connectome cache files with `git check-ignore`.
2. Check that ordinary source files and similarly named directories such as
   `development` and `device` remain unaffected by the new rules.
3. Verify that no indexed path matches the approved exclusion rules.
4. Compare unrelated index entries and the worktree state with the saved
   baseline; verify that all existing excluded files remain on disk.
5. Verify that each original branch and HEAD identity is unchanged.
6. Verify that each repository contains one managed exclusion block and that
   reapplying the rule transformation would not change the exclusion file.

Application tests, JAX compatibility checks, typing checks, and line-coverage
targets do not apply because no application code changes.

## Rules selected after inventory

The common managed block will contain these patterns:

```gitignore
dev/
dev */
connectome_cache/
connectome_data/
plos_runs/
full_runs/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.cache/
*.ncu-rep
*.nsys-rep
*.qdrep
/brainevent/_tcsr/正确性测试/无延迟前向/stabilized_model/
/brainevent/_tcsr/正确性测试/无延迟前向/计算正确性与梯度审查/验证结果/
```

`dev */` covers the existing `dev csr`, `dev dense`, `dev fcn`, and `dev jit`
directories. The two anchored paths contain generated model bundles and
validation outputs in the connectome checkout. Its nested `multi-area-model`
repository additionally excludes `/simulations/*/`, which covers generated
simulation directories while retaining `simulations/.gitignore` and source
files. Reference datasets and ordinary `.npy`/`.npz` fixtures remain trackable.

## Existing negation rules discovered during full validation

The training checkout's root `.gitignore` has explicit `!` exceptions that reopen
files below `brainevent/_tcsr/dev/Fly wire/braintrace`. These rules take precedence
over `info/exclude`; a complete `git ls-files --others --exclude-standard` check
found eighteen paths still eligible for snapshots after the initial update.

Back up that `.gitignore` and append a clearly marked `dev/` and `dev */` override
after the existing exceptions on its current non-main worktree branch. Preserve
the original content and index state. Validate all actual eligible untracked
paths in every repository, in addition to the synthetic path checks, so older
negations cannot silently defeat the exclusions.

## Execution result (2026-10-16)

All ten repositories now have the managed local exclusions. Existing branches
and HEADs are unchanged. Index-only removals were:

| Checkout | Removed entries |
| --- | ---: |
| `brainevent-dev` | 284 |
| `brainevent-连接组/brainevent` | 63 |
| `brainevent-mm进一步优化/brainevent-main` | 60 |
| `brainevent-训练与tcsr开发/brainevent` | 11 |
| `collaboration/BrainEvent-Perf-Benchmark` | 4 |
| Remaining five repositories | 0 |
| **Total** | **422** |

Validation passed for 201 positive path probes, 80 unaffected-path probes,
unrelated index entries, original branch/HEAD identities, and a complete scan
of eligible untracked files in every repository. All 329 excluded files that
existed before the index update remain on disk. The other removed entries
already referred to missing files. Existing Git object packs and checkpoint
references were not deleted; this change does not reclaim their disk space.

Backups and the machine-readable verification report are stored at:

`/home/luruheng/.local/state/brainevent-git-exclusions/20261016T161115-3843820/`

No commits or pushes were performed. The training checkout's `.gitignore`
override and this specification remain available for review without staging
them or changing unrelated pre-existing staged entries.

The new specification and the appended training `.gitignore` block pass
whitespace checks. A broader check of pre-existing benchmark working changes
reported unrelated whitespace in an existing Python edit and CSV output;
those user changes were preserved and excluded from this change's validation.
