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
import shutil
from pathlib import Path

from brainevent._version import __version__


class CompilationCache:
    """Persistent, filesystem-backed compilation cache.

    The cache key is a SHA-256 digest, truncated to the first 16 hex digits
    (64-bit collision domain), of:

    - User CUDA / C++ source code
    - ``brainevent`` version
    - ``jaxlib`` version (the FFI ABI moves with jaxlib)
    - nvcc / host-compiler version string
    - GPU architecture
    - Extra compiler / linker flags
    - Extra include paths (``-I`` search dirs change which headers win)
    - Byte hashes of the injected headers (``ffi_compat.h`` and jaxlib's
      ``ffi.h``), so an editable header edit or a jaxlib ABI bump rebuilds.

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
            Extra compiler / linker flags.  Hashed as a sorted JSON list.
        extra_include_paths : list of str, optional
            Additional ``-I`` header search paths.  These change which header a
            given ``#include`` resolves to, so two otherwise-identical builds
            with different include dirs must get **different** keys (a shadowing
            header would otherwise yield a wrong cached ``.so``).  Hashed as a
            sorted JSON list.
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
        h.update(source.encode())
        h.update(arch.encode())
        h.update(cxx_version.encode())
        h.update(__version__.encode())
        h.update(jaxlib_version.encode())
        h.update(json.dumps(extra_cflags or [], sort_keys=True).encode())
        h.update(json.dumps(extra_ldflags or [], sort_keys=True).encode())
        h.update(json.dumps(extra_include_paths or [], sort_keys=True).encode())
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

        The artefact is staged into a pid-suffixed temp file next to the
        destination, then ``os.replace``-d into place so concurrent readers
        never observe a partially written library.

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
        tmp = dest_dir / f".{name}.{os.getpid()}.tmp{self._ext()}"
        try:
            if source_is_user_dir:
                # User-owned artefact: copy so we never relocate the caller's file.
                shutil.copy2(src, tmp)
            else:
                try:
                    os.replace(src, tmp)        # same-filesystem atomic move
                except OSError:
                    shutil.copy2(src, tmp)      # cross-filesystem fallback
            os.replace(tmp, dest)               # atomic publish
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

        If *name* is given, only entries whose directory starts with
        ``<name>_`` are removed.
        """
        removed = 0
        if not self.base_dir.exists():
            return 0
        for entry in self.base_dir.iterdir():
            if name is not None and not entry.name.startswith(f"{name}_"):
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
            shutil.rmtree(entry, ignore_errors=True)
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
