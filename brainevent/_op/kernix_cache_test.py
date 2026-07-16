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

"""Tests for Kernix compilation cache internals."""

import hashlib
import os
import threading
from pathlib import Path

import pytest

from brainevent._op.kernix_cache import CompilationCache


# ---------------------------------------------------------------------------
# H4 — cache key must include extra_include_paths, header byte-contents,
#      and the jaxlib version.
# ---------------------------------------------------------------------------

def test_h4a_key_depends_on_extra_include_paths(tmp_path):
    """Two keys differing ONLY in ``extra_include_paths`` must differ.

    A shadowing header on a different ``-I`` path can change the compiled
    binary, so the include search path must be part of the cache key.
    """
    cache = CompilationCache(base_dir=str(tmp_path))
    common = dict(source="__global__ void k(){}", arch="sm_80")

    key_a = cache.cache_key(**common, extra_include_paths=["/opt/a/include"])
    key_b = cache.cache_key(**common, extra_include_paths=["/opt/b/include"])

    assert key_a != key_b


def test_h4b_key_depends_on_header_bytes(tmp_path):
    """Mutating the bytes of a hashed header file must change the key.

    ``__version__`` is only a proxy for the injected headers and does not move
    during an editable install, so the header *contents* must be hashed.
    """
    cache = CompilationCache(base_dir=str(tmp_path))
    header = tmp_path / "ffi_compat.h"
    header.write_text("#define ABI 1\n")
    common = dict(source="int main(){}", arch="cpu")

    key_before = cache.cache_key(**common, header_paths=[str(header)])
    header.write_text("#define ABI 2\n")  # struct layout / macro change
    key_after = cache.cache_key(**common, header_paths=[str(header)])

    assert key_before != key_after


def test_h4c_key_depends_on_jaxlib_version(monkeypatch, tmp_path):
    """A different ``jaxlib.__version__`` must change the key.

    ``pip install -U jaxlib`` changes the FFI ABI but reuses the cache dir; the
    jaxlib version must therefore participate in the key.
    """
    import jaxlib

    cache = CompilationCache(base_dir=str(tmp_path))
    common = dict(source="int main(){}", arch="cpu")

    monkeypatch.setattr(jaxlib, "__version__", "0.9.1", raising=False)
    key_old = cache.cache_key(**common)
    monkeypatch.setattr(jaxlib, "__version__", "0.10.0", raising=False)
    key_new = cache.cache_key(**common)

    assert key_old != key_new


def test_h4_defaults_backward_compatible(tmp_path):
    """The new parameters are optional: the legacy call signature still works."""
    cache = CompilationCache(base_dir=str(tmp_path))
    # Must not raise and must return a 16-hex key.
    key = cache.cache_key(
        source="int main(){}",
        arch="cpu",
        cxx_version="g++ 11",
        extra_cflags=["-O3"],
        extra_ldflags=["-lm"],
    )
    assert isinstance(key, str)
    assert len(key) == 16
    int(key, 16)  # valid hex


# ---------------------------------------------------------------------------
# M6 — clear() symlink handling; store() tmp leak / user-artifact move.
# ---------------------------------------------------------------------------

def test_m6a_clear_removes_and_counts_symlink_once(tmp_path):
    """A symlinked cache entry must be removed and counted exactly once.

    ``entry.is_dir()`` follows symlinks, so the old gate let a symlink through
    to ``shutil.rmtree`` (which raises on a symlink), leaving it on disk while
    still counting it as removed.
    """
    base = tmp_path / "cache"
    base.mkdir()
    cache = CompilationCache(base_dir=str(base))

    # A real entry + a symlink entry pointing outside the cache.  Entry names
    # must be exactly ``<name>_<16-hex-key>`` for clear() to match them.
    real_entry = base / "mod_0123456789abcdef"
    real_entry.mkdir()
    (real_entry / "mod.so").write_bytes(b"\x00")

    outside = tmp_path / "outside_target"
    outside.mkdir()
    (outside / "keep.txt").write_text("precious")
    link_entry = base / "mod_fedcba9876543210"
    link_entry.symlink_to(outside, target_is_directory=True)

    removed = cache.clear("mod")

    assert removed == 2, f"expected 2 real removals, got {removed}"
    assert not link_entry.exists() or not link_entry.is_symlink()
    assert not real_entry.exists()
    # The symlink target (a user dir) must survive — only the link is removed.
    assert outside.exists() and (outside / "keep.txt").read_text() == "precious"


def test_m6b_store_failure_leaves_no_tmp(tmp_path, monkeypatch):
    """If the atomic publish fails, ``store()`` must not leak a ``.tmp`` file."""
    base = tmp_path / "cache"
    cache = CompilationCache(base_dir=str(base))

    src = tmp_path / "build" / "mod.so"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"BINARY")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        # First call = move/stage into tmp (allow); second = publish (fail).
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("simulated publish failure")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(OSError):
        cache.store("mod", "key0000000000000", str(src))

    dest_dir = cache.cache_dir_for("mod", "key0000000000000")
    leftover = list(dest_dir.glob("*.tmp*")) if dest_dir.exists() else []
    assert leftover == [], f"leaked tmp files: {leftover}"


def test_m6c_store_does_not_relocate_user_build_dir(tmp_path):
    """``store()`` must COPY (not move) a user-supplied build artifact.

    With a user ``build_directory`` the source ``.so`` belongs to the caller;
    moving it silently relocates their file into the cache.
    """
    base = tmp_path / "cache"
    cache = CompilationCache(base_dir=str(base))

    user_build = tmp_path / "user_build"
    user_build.mkdir()
    src = user_build / "mod.so"
    src.write_bytes(b"USER_ARTIFACT")

    dest = cache.store(
        "mod", "key0000000000001", str(src), source_is_user_dir=True
    )

    assert Path(dest).exists()
    assert src.exists(), "store() deleted/relocated the caller's source file"
    assert src.read_bytes() == b"USER_ARTIFACT"


def test_m6c_store_may_move_internal_tmp_build(tmp_path):
    """For an internal (non-user) build dir, moving the source is fine.

    The default behaviour preserves the previous optimisation of moving the
    just-built artifact out of a throwaway build dir.
    """
    base = tmp_path / "cache"
    cache = CompilationCache(base_dir=str(base))

    src = tmp_path / "internal_build" / "mod.so"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"TMP_ARTIFACT")

    dest = cache.store("mod", "key0000000000002", str(src))

    assert Path(dest).exists()
    assert Path(dest).read_bytes() == b"TMP_ARTIFACT"


# ---------------------------------------------------------------------------
# Platform-specific cache publishing
# ---------------------------------------------------------------------------

def test_cache_uses_platform_ext(tmp_path, monkeypatch):
    from brainevent._op import kernix_toolchain as ktool

    monkeypatch.setattr(ktool.sys, "platform", "win32")
    cache = CompilationCache(base_dir=str(tmp_path))
    assert cache.lookup("m", "deadbeef") is None

    src = tmp_path / "src.dll"
    src.write_bytes(b"x")

    dest = cache.store("m", "deadbeef", str(src))

    assert dest.name == "m.dll"
    assert cache.lookup("m", "deadbeef") == dest


def test_cache_store_atomic_move(tmp_path):
    cache = CompilationCache(base_dir=str(tmp_path))
    build = tmp_path / "build"
    build.mkdir()
    so = build / "m.so"
    so.write_bytes(b"hello")

    dest = cache.store("m", "k", str(so))

    assert dest.read_bytes() == b"hello"
    assert not so.exists()


# ---------------------------------------------------------------------------
# L5 — docstring / truncation documentation (smoke checks).
# ---------------------------------------------------------------------------

def test_l5_key_is_64bit_truncated(tmp_path):
    """The published key is a 16-hex (64-bit) truncation of the SHA-256 digest."""
    cache = CompilationCache(base_dir=str(tmp_path))
    key = cache.cache_key(source="x", arch="cpu")
    assert len(key) == 16


def test_l5_docstring_no_longer_says_jax_kernel_bridge():
    """The stale 'jax-kernel-bridge version' wording must be gone."""
    assert "jax-kernel-bridge" not in (CompilationCache.__doc__ or "")


# ---------------------------------------------------------------------------
# F3 — resolved arg-specs must participate in the cache key.
# ---------------------------------------------------------------------------

def test_f3_key_depends_on_specs(tmp_path):
    """Same source, different ``specs`` → different keys.

    The compiled ``.so`` is user source PLUS generated FFI wrappers derived from
    the resolved specs, so a different ``functions`` mapping (different wrapper
    ABI) must not collide on the same cache entry.
    """
    cache = CompilationCache(base_dir=str(tmp_path))
    common = dict(source="void f(){}", arch="cpu")

    key_a = cache.cache_key(**common, specs='[{"name":"f","num_args":1}]')
    key_b = cache.cache_key(**common, specs='[{"name":"f","num_args":2}]')

    assert key_a != key_b
    # Absent specs stays backward-compatible and stable.
    assert cache.cache_key(**common) == cache.cache_key(**common)


def test_f3_extra_digests_change_key(tmp_path):
    """The ``extra_digests`` summary channel (capped header sets) affects the key."""
    cache = CompilationCache(base_dir=str(tmp_path))
    common = dict(source="x", arch="cpu")
    assert cache.cache_key(**common, extra_digests=["h1"]) != cache.cache_key(
        **common, extra_digests=["h2"]
    )


# ---------------------------------------------------------------------------
# F19 — clear(name) matches "<name>_<16 hex>" exactly (no prefix bleed).
# ---------------------------------------------------------------------------

def test_f19_clear_prefix_is_exact(tmp_path):
    """``clear("add")`` must not also delete ``add_fast_<key>`` siblings."""
    base = tmp_path / "cache"
    base.mkdir()
    cache = CompilationCache(base_dir=str(base))

    add = base / "add_0123456789abcdef"
    add.mkdir()
    (add / "add.so").write_bytes(b"\x00")
    add_fast = base / "add_fast_0123456789abcdef"
    add_fast.mkdir()
    (add_fast / "add_fast.so").write_bytes(b"\x00")

    removed = cache.clear("add")

    assert removed == 1, f"clear('add') should remove only 'add_*', got {removed}"
    assert not add.exists()
    assert add_fast.exists(), "clear('add') wrongly deleted add_fast_* sibling"


def test_f19_clear_ignores_non_key_dirs(tmp_path):
    """A directory not shaped like ``<name>_<16hex>`` is left untouched."""
    base = tmp_path / "cache"
    base.mkdir()
    cache = CompilationCache(base_dir=str(base))
    # 15 hex chars — not a valid key length.
    bad = base / "add_0123456789abcde"
    bad.mkdir()
    assert cache.clear("add") == 0
    assert bad.exists()


# ---------------------------------------------------------------------------
# F15 — store() tolerates a Windows-locked DLL (dest already published).
# ---------------------------------------------------------------------------

def test_f15_store_returns_dest_when_replace_permission_denied(tmp_path, monkeypatch):
    """A PermissionError on publish is benign iff ``dest`` already exists.

    Simulates the Windows "os.replace over a loaded DLL" failure: the first
    publish succeeds, the second raises PermissionError but a valid artefact is
    already present, so store() returns it instead of propagating (finding 15).
    """
    base = tmp_path / "cache"
    cache = CompilationCache(base_dir=str(base))

    # First store publishes dest normally.
    src1 = tmp_path / "b1" / "m.so"
    src1.parent.mkdir(parents=True)
    src1.write_bytes(b"PUBLISHED")
    dest = cache.store("m", "0123456789abcdef", str(src1))
    assert Path(dest).read_bytes() == b"PUBLISHED"

    # Now make os.replace raise PermissionError only on the *publish* step.
    real_replace = os.replace

    def locked_replace(a, b):
        if str(b) == str(dest):
            raise PermissionError("[WinError 5] Access is denied (DLL locked)")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", locked_replace)

    src2 = tmp_path / "b2" / "m.so"
    src2.parent.mkdir(parents=True)
    src2.write_bytes(b"NEWER")
    # dest exists → treated as already-published → returns dest, no raise.
    out = cache.store("m", "0123456789abcdef", str(src2))
    assert Path(out) == Path(dest)
    assert Path(dest).read_bytes() == b"PUBLISHED"  # kept the published one
    # No leaked tmp file.
    assert not list(Path(dest).parent.glob("*.tmp*"))


def test_f15_store_reraises_when_dest_absent(tmp_path, monkeypatch):
    """A PermissionError with no existing ``dest`` is a real failure → re-raise."""
    base = tmp_path / "cache"
    cache = CompilationCache(base_dir=str(base))
    src = tmp_path / "b" / "m.so"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"X")

    real_replace = os.replace

    def deny_publish(a, b):
        # allow staging into tmp, deny the publish (dest never created)
        name = os.path.basename(str(b))
        if name.endswith(".so") and ".tmp" not in name:
            raise PermissionError("Access is denied")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", deny_publish)
    with pytest.raises(PermissionError):
        cache.store("m", "fedcba9876543210", str(src))


# ---------------------------------------------------------------------------
# F11 — concurrent same-process store() must publish a valid (uncorrupted) .so.
# ---------------------------------------------------------------------------

def test_f11_concurrent_store_publishes_valid_artifact(tmp_path):
    """Two threads storing the same name/key must not publish a truncated lib.

    Both threads (same pid) previously shared a single ``.tmp`` staging path,
    so one could ``os.replace`` a file the other was still writing.  With a
    uuid-suffixed temp name the published artefact must byte-match the payload.
    """
    base = tmp_path / "cache"
    cache = CompilationCache(base_dir=str(base))

    # A multi-KB payload so a torn write would be detectable via the hash.
    payload = (b"KERNEL-BYTES-" * 4096)
    expected = hashlib.sha256(payload).hexdigest()

    n = 8
    barrier = threading.Barrier(n)
    errors: list[Exception] = []

    def worker(i: int):
        # Each thread owns a distinct source file with identical content.
        src = tmp_path / f"build{i}" / "cc.so"
        src.parent.mkdir(parents=True)
        src.write_bytes(payload)
        try:
            barrier.wait()
            cache.store("cc", "0123456789abcdef", str(src))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent store raised: {errors}"
    dest = cache.cache_dir_for("cc", "0123456789abcdef") / "cc.so"
    assert dest.exists()
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert got == expected, "published artefact is corrupt/truncated"
    # No staging temp files left behind.
    assert not list(dest.parent.glob("*.tmp*"))
