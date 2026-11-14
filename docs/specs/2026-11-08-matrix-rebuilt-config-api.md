# Explicit matrix reconstruction with rebuilt configuration

## Goal

Name the public operation so callers can distinguish configuration generation
from returning a matrix with rebuilt configuration.

## Decisions

- Rename `prepare_config` to `with_rebuilt_config` for CSR/CSC and FixedNum
  matrices. Remove the old name without a compatibility alias.
- Preserve behavior: rebuild structure and device plans using the existing
  tuning snapshot, then return a new matrix sharing the original arrays and
  structure caches. Do not mutate the source or reload persisted tuning.
- An explicit device selects configuration preparation; it does not migrate
  matrix values or index arrays. Require concrete structure outside JIT.
- Keep configuration-only helpers and matrix construction paths unchanged.
  This rename does not change delaycsr's inherited return behavior.
- Update current API documentation and callers; retain dated design records.
  Delay delivery uses its own metadata and workspace, not CSRConfig data;
  retaining config for inheritance and pytree handling remains unchanged.

## Steps

1. Rename both methods and clarify their NumPy-style docstrings.
2. Update current documentation, tests, embedded scripts and error messages.
3. Verify source immutability, array sharing and rebuilt plans for both families.

## Verification

Run focused CSR/FCN configuration lifecycle, traced-structure, backend and
device-placement tests. Check the public rename, repository typing and diff
whitespace. Record CPU/GPU validation limits without changing other APIs.
