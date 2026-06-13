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

"""Reproduction tests for the cache audit findings H4, M6 and L5.

These tests are hermetic: they exercise ``CompilationCache`` directly (the
cache key is a pure function; the filesystem operations use ``tmp_path``) and
do **not** require nvcc, a host compiler, or a GPU.  Run with::

    python -m pytest brainevent/_op/kernix_cache_audit_test.py -m "" -q
"""

import os
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


def test_h4_cache_header_paths_covers_every_brainevent_header():
    """``_cache_header_paths`` hashes *all* injected headers, not just ffi_compat.h.

    A semantics change in ``check.h`` (abort -> throw), ``tensor.h`` (bounds
    guards) or ``cuda_common.h`` must rebuild even when the ``brainevent``
    version string is unchanged (editable installs), so the pipeline must feed
    every brainevent header — both the ``brainevent/`` subdir and the top-level
    headers — into the cache key.
    """
    import types
    import brainevent
    from brainevent._op.kernix_pipeline import _cache_header_paths

    be_inc = os.path.join(os.path.dirname(brainevent.__file__), "include")
    # ``xla_ffi_include_dir`` points at a non-existent tree: ffi.h is simply
    # skipped (missing files are tolerated), isolating the brainevent coverage.
    toolchain = types.SimpleNamespace(
        brainevent_include_dir=be_inc,
        xla_ffi_include_dir="/nonexistent-xla-include",
    )

    paths = _cache_header_paths(toolchain)
    names = {os.path.basename(p) for p in paths}

    # Headers from the ``brainevent/`` subdir and the top-level include root.
    for required in ("ffi_compat.h", "check.h", "tensor.h", "dtypes.h", "cuda_common.h"):
        assert required in names, f"{required} missing from cache header set: {names}"
    # Every returned path must actually exist (missing entries are filtered out).
    for path in paths:
        assert os.path.isfile(path)


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

    # A real entry + a symlink entry pointing outside the cache.
    real_entry = base / "mod_realkey0000000"
    real_entry.mkdir()
    (real_entry / "mod.so").write_bytes(b"\x00")

    outside = tmp_path / "outside_target"
    outside.mkdir()
    (outside / "keep.txt").write_text("precious")
    link_entry = base / "mod_linkkey0000000"
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
