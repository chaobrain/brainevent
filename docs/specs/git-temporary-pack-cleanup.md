# Remove audited Git temporary-pack garbage

## Approved scope

Remove only the 166 temporary pack files previously audited in:

`/home/luruheng/BrainEvent/brainevent-连接组/brainevent/.git/objects/pack/`

The audit manifest is:

`/home/luruheng/.local/state/brainevent-git-exclusions/20261016T161115-3843820/git-cache-audit.json`

Their total allocated size is 263,778,889,728 bytes (about 245.66 GiB).
The user authorized clearing Git garbage first. Retain both formal pack/index
pairs, the Codex checkpoint reference, all loose objects, the current index,
the exclusion settings, and every working-tree file. Do not commit or push.

## Preconditions

1. Confirm the current worktree branches are not `main` or `master`.
2. Confirm `git count-objects -v` identifies exactly the manifest's 166 files
   as garbage, with matching total allocated size.
3. Require every manifest path to be a regular file directly inside the
   approved pack directory, named `tmp_pack_` followed by six alphanumeric
   characters. Check its size and modification time against the prior audit.
4. Confirm no Git process owned by the current user is running and no process
   visible to this user has the pack directory's files open.
5. Save branch/HEAD identities, all refs, the index checksum, formal pack-file
   metadata, and the exact removal manifest before deleting files.

## Execution and verification

Delete only the enumerated, revalidated temporary files. Record each removal
and stop on an unexpected state change. Avoid blanket Git object deletion,
reflog expiration, pruning, and removal of checkpoint refs.

After cleanup, require:

- All 166 enumerated temporary files are absent and the garbage count is zero.
- Both formal packs and their sidecars retain their original metadata.
- Branches, HEADs, all refs, the index checksum, and exclusion settings remain
  unchanged.
- `git fsck --connectivity-only --no-dangling` succeeds; this checks object
  reachability and existence without decoding all large cached blobs.
- The real connectome cache is still present on disk and excluded from Git.

Record the before/after filesystem availability and a machine-readable cleanup
report alongside the audit. Edge cases include newly active Git processes,
replaced files or symlinks, changing file timestamps, partial deletion, and
unrelated concurrent disk activity. Abort on manifest mismatches; report actual
filesystem measurements separately from the removed files' allocated size.

## Result (2026-10-16)

Removed exactly the 166 audited temporary files, totaling 263,778,889,728
allocated bytes (245.66 GiB). The Git garbage count is now zero. Root filesystem
availability increased from approximately 79.7 GiB to 325.4 GiB.

Both formal packs and their sidecars, all loose-object counts, the branch/HEAD,
all refs (including the Codex checkpoint), the current index checksum, and
the exclusion file checksum are unchanged. The post-cleanup connectivity check
passed, and the real connectome cache remains present and ignored.

No commits or pushes were performed. The cleanup manifest, per-file deletion
journal, and verification report are under:

`/home/luruheng/.local/state/brainevent-git-exclusions/20261016T161115-3843820/git-garbage-cleanup-20261016T164155/`
