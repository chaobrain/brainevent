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

    The cache key is a SHA-256 digest of:
    - User CUDA source code
    - jax-kernel-bridge version
    - nvcc version string
    - GPU architecture
    - Extra compiler / linker flags

    Cached artefacts are stored under
    ``<base_dir>/<name>_<key>/module.so``.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(
            base_dir or
            os.environ.get("BRAINEVENT_CACHE_DIR", Path.home() / ".cache" / "brainevent")
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
    ) -> str:
        """Compute a deterministic cache key."""
        h = hashlib.sha256()
        h.update(source.encode())
        h.update(arch.encode())
        h.update(cxx_version.encode())
        h.update(__version__.encode())
        h.update(json.dumps(extra_cflags or [], sort_keys=True).encode())
        h.update(json.dumps(extra_ldflags or [], sort_keys=True).encode())
        return h.hexdigest()[:16]

    def cache_dir_for(self, name: str, key: str) -> Path:
        return self.base_dir / f"{name}_{key}"

    # ------------------------------------------------------------------

    def lookup(self, name: str, key: str) -> Path | None:
        """Return the .so path if the cache entry exists, else None."""
        so_path = self.cache_dir_for(name, key) / f"{name}.so"
        if so_path.exists():
            return so_path
        return None

    def store(self, name: str, key: str, so_path: str) -> Path:
        """Copy a built .so into the cache and return the cached path."""
        dest_dir = self.cache_dir_for(name, key)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.so"
        if str(Path(so_path).resolve()) != str(dest.resolve()):
            shutil.copy2(so_path, dest)
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
            if not entry.is_dir():
                continue
            if name is not None and not entry.name.startswith(f"{name}_"):
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
