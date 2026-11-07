# CSR configuration layout

## Goal

Keep device properties, hybrid MV tuning, CSR structure statistics, and hybrid MM planning at their respective lifetimes. Move CSR configuration preparation out of `main.py` while preserving existing saved tuning files and callers.

## Decisions

- `CSRConfig` owns an optional device snapshot, an `MVSchedule`, common structure fields (`shape`, `nnz`, and longest row for each traversal), and optional primary/mirror MM structure plans. MV and MM share the structure fields without making MV depend on MM-specific masks.
- `MVSchedule` owns the four persisted CUDA scheduler constants and their compiler flags, module suffix, and task-capacity calculation. Keep the existing four-argument `CSRConfig` constructor and flat read-only accessors as compatibility aliases.
- An MM structure plan owns geometry and the long-row mask for one traversal. `MMCallPlan` remains a derived, cached value for a particular batch, dtype, and weighted mode; it is not stored or persisted.
- Save/load in `binary/initialize.py` continues to use the existing flat four-field JSON and environment schema. Device and structure data are prepared from the concrete matrix and target device, never loaded from tuning persistence.
- Preserve JAX's separation of hashable primitive metadata and array mask leaves. Rebuild or invalidate prepared structure data when traversal layout or structure changes.

## Steps

1. Refactor `config.py` data classes, planner methods, and configuration preparation helpers.
2. Move the helper imports in CSR/FCN code, then update MV/MM call sites and tuning persistence to use the new fields.
3. Update focused lifecycle, integration, and PyTree tests; verify legacy construction and JSON loading.

## Verification

Run the focused CSR configuration, initialization, integration, lifecycle, and FCN configuration tests. Check typing if the repository's mypy setup permits it. GPU-only kernels require a GPU environment for runtime verification.
