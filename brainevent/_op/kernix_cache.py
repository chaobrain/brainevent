# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Source-hash-based compilation cache."""

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from brainevent._version import __version__

# Bumped whenever the *layout* of ``cache_key`` inputs changes (not the source
# content).  Folded into every digest so a schema change (e.g. adding the
# resolved ``specs`` in finding 3) yields wholesale-different keys rather than
# silently colliding with, or half-matching, cache dirs written by an older
# brainevent that hashed a different set of inputs.
_KEY_SCHEMA = "2"

# A cache entry directory is ``<name>_<16-hex-key>``.  Matching the key shape
# exactly (rather than a bare ``<name>_`` prefix) keeps ``clear("add")`` from
# also deleting sibling entries such as ``add_fast_<key>`` (finding 19).
_CACHE_KEY_RE = "[0-9a-f]{16}"


class CompilationCache:
    """Persistent, filesystem-backed compilation cache.

    The cache key is a SHA-256 digest, truncated to the first 16 hex digits
    (64-bit collision domain), of:

    - User CUDA / C++ source code
    - The resolved function arg-specs (drive the generated wrapper ABI baked
      into the ``.so`` — the same source with a different ``functions`` mapping
      compiles to a different artefact; finding 3)
    - ``brainevent`` version
    - ``jaxlib`` version (the FFI ABI moves with jaxlib)
    - nvcc / host-compiler version string
    - GPU architecture
    - Extra compiler / linker flags (order preserved — order is significant)
    - Extra include paths (``-I`` search dirs change which headers win)
    - Byte hashes of the injected headers (``ffi_compat.h``, jaxlib's ``ffi.h``,
      and any ``*.h``/``*.cuh``/``*.hpp`` found under ``extra_include_paths``),
      so an editable header edit or a jaxlib ABI bump rebuilds.

    Cached artefacts are stored under
    ``<base_dir>/<name>_<key>/module.so``.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(
            base_dir
            or os.environ.get("BRAINEVENT_CACHE_DIR")
            or str(Path.home() / ".cache" / "brainevent" / __version__)
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def cache_key(
        self,
        source: str,
        arch: str,
        cxx_version: str = "",
        extra_cflags: list[str] | None = None,
        extra_ldflags: list[str] | None = None,
        extra_include_paths: list[str] | None = None,
        header_paths: list[str] | None = None,
        jaxlib_version: str | None = None,
        specs: str = "",
        extra_digests: list[str] | None = None,
    ) -> str:
        """Compute a deterministic cache key for a compiled artefact.

        Parameters
        ----------
        source : str
            Preprocessed/user source code that will be compiled.
        arch : str
            Target architecture token(s) (e.g. ``"sm_86+sm_90"`` or ``"cpu"``).
        cxx_version : str, optional
            Host-compiler / nvcc version string.  Two compilers with different
            versions must not share a cache entry.
        extra_cflags, extra_ldflags : list of str, optional
            Extra compiler / linker flags.  Serialized as a JSON list with
            **order preserved** — flag order is semantically significant
            (``-I`` search precedence, last-flag-wins overrides), so the list is
            hashed as given and is *not* sorted (``json.dumps(..., sort_keys=True)``
            only orders dict keys, never list elements).
        extra_include_paths : list of str, optional
            Additional ``-I`` header search paths.  These change which header a
            given ``#include`` resolves to, so two otherwise-identical builds
            with different include dirs must get **different** keys (a shadowing
            header would otherwise yield a wrong cached ``.so``).  Serialized as
            a JSON list with **order preserved** (search precedence matters).
        header_paths : list of str, optional
            Filesystem paths of injected headers whose **byte contents** affect
            the build (``brainevent``'s ``ffi_compat.h`` and jaxlib's
            ``xla/ffi/api/ffi.h``).  Each file is SHA-256-hashed so an editable
            header edit, or a jaxlib upgrade that rewrites ``ffi.h``, forces a
            rebuild.  Missing/unreadable files contribute a sentinel rather than
            raising, so key computation never fails on a transient read error.
        jaxlib_version : str, optional
            ``jaxlib.__version__``.  The FFI ABI moves with jaxlib, so a jaxlib
            upgrade (same ``brainevent`` version, same source) must rebuild.
            Imported lazily when ``None`` so the key always reflects the
            installed jaxlib; pass an explicit value to override (e.g. tests).
        specs : str, optional
            Canonical serialization of the resolved function arg-specs
            (finding 3).  The compiled artefact is ``user_source`` **plus**
            generated FFI wrappers derived from these specs, so the same source
            with a different ``functions`` mapping produces a different ``.so``
            ABI and must therefore get a different key.  Hashed verbatim.
        extra_digests : list of str, optional
            Pre-computed digest strings for inputs the caller summarises itself
            (e.g. the names+mtimes of a very large ``extra_include_paths``
            header set that would be too expensive to byte-hash — finding 10).
            Hashed as a JSON list with order preserved.

        Returns
        -------
        str
            The first 16 hex digits (64-bit truncation) of the SHA-256 digest.
            64 bits is ample for a per-machine on-disk cache; the truncation
            keeps directory names short.  Collisions are astronomically
            unlikely but theoretically possible within this 64-bit domain.

        Notes
        -----
        New parameters are optional with backward-compatible defaults so legacy
        callers keep working; an omitted input simply contributes its empty
        sentinel to the digest.
        """
        if jaxlib_version is None:
            try:
                import jaxlib
                jaxlib_version = jaxlib.__version__
            except Exception:
                jaxlib_version = ""

        h = hashlib.sha256()
        h.update(_KEY_SCHEMA.encode())
        h.update(source.encode())
        h.update(arch.encode())
        h.update(cxx_version.encode())
        h.update(__version__.encode())
        h.update(jaxlib_version.encode())
        # Resolved arg-specs drive the generated wrapper ABI baked into the .so
        # (finding 3): the same source with different `functions` must not share
        # a cache entry.
        h.update(specs.encode())
        h.update(json.dumps(extra_cflags or [], sort_keys=True).encode())
        h.update(json.dumps(extra_ldflags or [], sort_keys=True).encode())
        h.update(json.dumps(extra_include_paths or [], sort_keys=True).encode())
        h.update(json.dumps(extra_digests or [], sort_keys=True).encode())
        # Byte-hash each injected header (sorted for determinism) so the key
        # tracks header *contents*, not just the brainevent version proxy.
        for path in sorted(header_paths or []):
            h.update(path.encode())
            try:
                h.update(hashlib.sha256(Path(path).read_bytes()).hexdigest().encode())
            except OSError:
                h.update(b"<unreadable>")
        # Truncate to 64 bits; see the Returns/Notes above for the rationale.
        return h.hexdigest()[:16]

    def cache_dir_for(self, name: str, key: str) -> Path:
        return self.base_dir / f"{name}_{key}"

    def _ext(self) -> str:
        """Shared-library extension for the current OS (.so/.dylib/.dll)."""
        from .kernix_toolchain import so_ext
        return so_ext()

    # ------------------------------------------------------------------

    def lookup(self, name: str, key: str) -> Path | None:
        """Return the shared-lib path if the cache entry exists, else None."""
        so_path = self.cache_dir_for(name, key) / f"{name}{self._ext()}"
        if so_path.exists():
            return so_path
        return None

    def store(
        self,
        name: str,
        key: str,
        so_path: str,
        source_is_user_dir: bool = False,
    ) -> Path:
        """Atomically publish a built shared lib into the cache.

        The artefact is staged into a temp file next to the destination whose
        name carries both the pid (for debuggability) **and** a random
        ``uuid4`` (finding 11) so two threads of the *same* process — sharing a
        pid — storing the same ``name``+``key`` never collide on the staging
        path and atomically publish each other's half-written library.  The
        staged file is then ``os.replace``-d into place so concurrent readers
        never observe a partial library.

        Parameters
        ----------
        name : str
            Module name; the published file is ``<name><ext>``.
        key : str
            Cache key identifying the destination directory.
        so_path : str
            Path to the freshly built shared library to publish.
        source_is_user_dir : bool, optional
            When ``True``, *so_path* lives in a caller-supplied
            ``build_directory`` and therefore belongs to the user: the source
            is **copied** rather than moved, so the caller's artefact is left
            in place.  When ``False`` (default) the source is a throwaway
            internal build dir and may be moved.

        Returns
        -------
        Path
            Path to the published shared library inside the cache.

        Notes
        -----
        The staging temp file is removed via ``try/finally`` if the final
        publish (or staging) raises, so a failed ``store`` never leaks a
        ``.tmp`` artefact into the cache directory.
        """
        dest_dir = self.cache_dir_for(name, key)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}{self._ext()}"
        src = Path(so_path).resolve()
        if str(src) == str(dest.resolve()):
            return dest
        # pid + uuid4: pid stays for debuggability, uuid4 guarantees uniqueness
        # across same-process threads and same-pid containers (finding 11).
        tmp = dest_dir / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp{self._ext()}"
        try:
            if source_is_user_dir:
                # User-owned artefact: copy so we never relocate the caller's file.
                shutil.copy2(src, tmp)
            else:
                try:
                    os.replace(src, tmp)        # same-filesystem atomic move
                except OSError:
                    shutil.copy2(src, tmp)      # cross-filesystem fallback
            try:
                os.replace(tmp, dest)           # atomic publish
            except PermissionError:
                # Windows locks a loaded DLL: os.replace over an existing,
                # mapped `dest` raises PermissionError (finding 15).  If a valid
                # artefact is already published at `dest`, that build is as good
                # as ours — return it (the staged tmp is dropped in `finally`).
                # If `dest` does not exist, the failure is real: re-raise.
                if dest.exists():
                    return dest
                raise
        finally:
            # If staging or publish failed, drop the temp file (best-effort).
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return dest

    # ------------------------------------------------------------------

    def clear(self, name: str | None = None) -> int:
        """Remove cached artefacts.  Returns number of entries removed.

        Parameters
        ----------
        name : str, optional
            If given, only entries whose directory name is exactly
            ``<name>_<16-hex-key>`` are removed.  Matching the full key shape
            (rather than a bare ``<name>_`` prefix) means ``clear("add")`` does
            **not** also delete unrelated siblings like ``add_fast_<key>``
            (finding 19).

        Returns
        -------
        int
            The number of entries actually removed.  An entry that fails to
            delete (e.g. a Windows DLL still mapped by another process — finding
            15) is **not** counted, so the return value never over-reports.
        """
        removed = 0
        if not self.base_dir.exists():
            return 0
        pattern = (
            re.compile(re.escape(name) + "_" + _CACHE_KEY_RE + r"\Z")
            if name is not None else None
        )
        for entry in self.base_dir.iterdir():
            if pattern is not None and not pattern.match(entry.name):
                continue
            # A symlink would pass ``is_dir()`` (it follows the link), but
            # ``shutil.rmtree`` raises on a symlink.  Unlink the link itself
            # (never its target) and count it as a real removal.
            if entry.is_symlink():
                try:
                    entry.unlink()
                except OSError:
                    continue
                removed += 1
                continue
            if not entry.is_dir():
                continue
            # ignore_errors=False + explicit existence check: a locked DLL on
            # Windows leaves the dir in place, and we must not count it removed.
            try:
                shutil.rmtree(entry, ignore_errors=False)
            except OSError:
                pass
            if not entry.exists():
                removed += 1
        return removed

    def size(self) -> tuple[int, int]:
        """Return ``(num_entries, total_bytes)``."""
        entries = 0
        total = 0
        if not self.base_dir.exists():
            return 0, 0
        for entry in self.base_dir.iterdir():
            if entry.is_dir():
                entries += 1
                for f in entry.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return entries, total
