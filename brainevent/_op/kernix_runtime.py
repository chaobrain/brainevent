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

"""Runtime layer: CompiledModule and JAX FFI registration."""

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any
import re
import threading

import jax

from brainevent._error import KernelError, KernelLoadError, KernelRegistrationError


def _missing_artifact_message(so_path: str) -> str:
    """Message for a cached artefact that vanished before it could be loaded.

    If the ``.so`` disappears between ``lookup()`` and ``CDLL`` (e.g. another
    process ran ``clear_cache``), the POSIX loader says "cannot open shared
    object file", which would otherwise match the cu12 "missing CUDA runtime"
    hint in :func:`_format_load_error` and misdirect the user (finding 19).
    """
    return (
        "[brainevent] compiled artefact is missing  (code=E-LOAD-MISSING)\n"
        "\n"
        f"Reason: the cached shared library no longer exists: {so_path}\n"
        "\n"
        "How to fix:\n"
        "  The artefact was present when the cache was looked up but is gone\n"
        "  now -- the cache entry was most likely cleared concurrently\n"
        "  (clear_cache(), or another process/user removing the cache dir).\n"
        "  Re-run to rebuild it; pass force_rebuild=True if it recurs."
    )


def _format_load_error(so_path: str, err: Exception) -> str:
    """Build an actionable message for a failed shared-library load.

    The heuristics inspect the platform loader's error text and append a
    tailored remediation hint.  Both POSIX ``dlopen`` wording and the
    Windows loader's ``LoadLibrary``/``FormatMessage`` phrasings are
    recognised so the hint is useful on either platform.

    Parameters
    ----------
    so_path : str
        Path to the artefact that failed to load.
    err : Exception
        The exception raised by :class:`ctypes.CDLL` (an :class:`OSError`,
        e.g. ``OSError("libcudart.so.12: cannot open shared object file ...")``
        on POSIX or ``OSError("[WinError 126] The specified module could not
        be found")`` on Windows).

    Returns
    -------
    str
        A multi-line, human-readable diagnostic ending with a "How to fix"
        section.

    Notes
    -----
    Windows loader phrasings are detected by both their ``FormatMessage`` text
    ("The specified module could not be found", "is not a valid Win32
    application") and their numeric ``WinError`` codes (126 = ``MOD_NOT_FOUND``,
    127 = ``PROC_NOT_FOUND``, 193 = ``BAD_EXE_FORMAT``), since ctypes may
    surface either form.
    """
    msg = str(err)
    low = msg.lower()

    # --- Windows loader signatures -------------------------------------
    # Numeric WinError codes (matched on word boundaries so "126" inside an
    # unrelated number does not trigger a false positive).
    win_codes = set(re.findall(r"\b(126|127|193)\b", low))
    win_mod_not_found = (
        "the specified module could not be found" in low
        or "the specified procedure could not be found" in low
        or "while loading dependent" in low
        or bool(win_codes & {"126", "127"})
    )
    win_bad_format = (
        "is not a valid win32 application" in low
        or "%1 is not a valid" in low
        or "193" in win_codes
    )

    lines = [
        "[brainevent GPU toolchain] Failed to load shared library  (code=E-LOAD)",
        "",
        f"Reason: cannot load the compiled artefact: {so_path}",
        f"loader: {msg}",
        "",
        "How to fix:",
    ]
    if "insufficient" in low or "forward compatibility" in low or "driver version" in low:
        lines += [
            "  1) The NVIDIA driver is too old for this CUDA toolkit. Upgrade the driver,",
            "     or install a jax[cudaNN] whose CUDA version matches your driver.",
        ]
    elif win_bad_format:
        lines += [
            "  1) Architecture/bitness mismatch: the DLL is not a valid Win32 application for",
            "     this process. Ensure a 64-bit Python loads a 64-bit (x64) build -- do not mix",
            "     32-bit and 64-bit toolchains.",
            "  2) Rebuild the artefact with the host compiler matching your Python's architecture.",
        ]
    elif win_mod_not_found:
        lines += [
            "  1) A dependent DLL could not be found by the Windows loader. The artefact itself",
            "     may load, but one of its dependencies (e.g. the CUDA runtime cudart64_*.dll,",
            "     or the MSVC runtime) is missing or not on the search path.",
            "  2) Add the directory holding the dependent DLLs to PATH (or use",
            "     os.add_dll_directory), and confirm jax[cuda*] / the CUDA toolkit is installed.",
            "  3) Inspect dependencies with a tool such as dumpbin /dependents or Dependencies.exe.",
        ]
    elif "cudart" in low or "cannot open shared object" in low or "no such file" in low:
        lines += [
            "  1) Missing CUDA runtime libraries (typically cu12). Ensure jax[cuda*] is installed correctly.",
            "  2) Add the CUDA runtime library directory to LD_LIBRARY_PATH (e.g. site-packages/nvidia/cuda_runtime/lib).",
        ]
    else:
        lines += ["  1) Verify the build succeeded and dependent libraries are available; set BRAINEVENT_TOOLCHAIN_DEBUG=1 to see a toolchain snapshot."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CompiledModule
# ---------------------------------------------------------------------------

class CompiledModule:
    """A compiled module loaded from a shared library.

    Each function listed at compilation time has a corresponding
    ``extern "C"`` symbol ``be_<name>`` in the ``.so``.  These are
    loaded via *ctypes* and wrapped for use with the JAX FFI system.

    Parameters
    ----------
    so_path : str
        Path to the compiled ``.so`` shared library.
    function_names : list[str]
        User function names whose FFI handler symbols to resolve.
    """

    def __init__(self, so_path: str, function_names: list[str]):
        self._so_path = str(so_path)
        self._content_hash: str | None = None
        try:
            self._lib = ctypes.CDLL(self._so_path)
        except OSError as e:
            # A vanished artefact gets a dedicated hint rather than the cu12
            # "missing CUDA runtime" misattribution (finding 19).
            if not os.path.exists(self._so_path):
                raise KernelLoadError(_missing_artifact_message(self._so_path)) from e
            raise KernelLoadError(_format_load_error(self._so_path, e)) from e
        # Values are ctypes function pointers (``ctypes._CFuncPtr`` is not a
        # public/typeshed name, so ``Any`` stands in for it).
        self._functions: dict[str, Any] = {}

        for fname in function_names:
            symbol = f"be_{fname}"
            try:
                fn = getattr(self._lib, symbol)
            except AttributeError:
                raise KernelError(
                    f"Symbol '{symbol}' not found in {so_path}. "
                    f"Available symbols may not include the FFI wrapper for "
                    f"'{fname}'. Did the compilation succeed?"
                )
            # XLA FFI handler signature: XLA_FFI_Error*(XLA_FFI_CallFrame*)
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
            self._functions[fname] = fn

    def get_handler(self, name: str):
        """Return the ctypes function pointer for an FFI handler.

        Parameters
        ----------
        name : str
            User function name (without the ``be_`` prefix).
        """
        if name not in self._functions:
            raise KeyError(
                f"Function '{name}' not found in module. "
                f"Available: {list(self._functions)}"
            )
        return self._functions[name]

    @property
    def path(self) -> str:
        """Path to the loaded ``.so``."""
        return self._so_path

    @property
    def content_hash(self) -> str:
        """SHA-256 (first 16 hex) of the loaded ``.so`` **bytes**.

        This is the *content* identity of the artefact, used as part of the FFI
        registration key (finding 4): two libraries built from different source
        (or with a different wrapper ABI) at the same cache path have different
        content hashes, so re-registration is not mistaken for an idempotent
        no-op.  Computed lazily and memoised — the file is read at most once per
        module (the "cold path"); an unreadable file degrades to a path-based
        sentinel rather than raising.
        """
        if self._content_hash is None:
            try:
                digest = hashlib.sha256(Path(self._so_path).read_bytes()).hexdigest()
                self._content_hash = "sha256:" + digest[:16]
            except OSError:
                self._content_hash = "path:" + self._so_path
        return self._content_hash

    @property
    def function_names(self) -> list[str]:
        """Names of available functions."""
        return list(self._functions)

    def __repr__(self) -> str:
        return f"CompiledModule(path={self._so_path!r}, functions={self.function_names})"


# ---------------------------------------------------------------------------
# JAX FFI registration bridge
# ---------------------------------------------------------------------------

# Global registry of (target_name → list of CompiledModule keep-alives) that
# prevents garbage collection of the ctypes CDLL while the FFI target is alive.
# It is a *list* (finding 4): a ``replace=True`` re-registration appends the new
# module and NEVER drops (or dlcloses) an old one, because a previously-compiled
# XLA executable may still hold a pointer into the earlier image.
_LIVE_MODULES: dict[str, list["CompiledModule"]] = {}

# Track registered names to give clear errors on duplicates.
_REGISTERED_TARGETS: set[str] = set()

# Identity of each registration, used to decide whether a same-name
# re-registration is an idempotent no-op (equivalent module) or a conflicting
# clobber (different content).  Maps target_name → (content_id, func_name,
# platform) where content_id is the ``.so`` byte hash (finding 4), so editing
# the kernel source — same path, different bytes — is correctly detected as a
# different registration rather than a stale no-op.
_REGISTRATION_KEYS: dict[str, tuple[str, str, str]] = {}

# Serialises the check-and-register sequence below.  ``jax.ffi.register_ffi_target``
# *silently overwrites* an existing target, so without this lock two threads can
# both pass the membership check and double-register (one of them clobbering the
# other's live module, dropping a still-referenced keep-alive).  Guarding the
# whole read-modify-write of ``_REGISTERED_TARGETS`` / ``_LIVE_MODULES`` /
# ``_REGISTRATION_KEYS`` makes registration atomic.
_REGISTRATION_LOCK = threading.Lock()


def _module_content_id(module: "CompiledModule") -> str:
    """Return a *content* identity for a module (finding 4).

    Prefers the module's ``content_hash`` (a hash of the ``.so`` bytes), so two
    artefacts that share a cache *path* but differ in bytes are distinguished.
    Falls back to the module's ``path`` for lightweight test doubles that do not
    expose ``content_hash`` (or whose file does not exist).
    """
    content = getattr(module, "content_hash", None)
    if content is not None:
        return str(content)
    return str(getattr(module, "path", module))


def _registration_key(
    module: "CompiledModule", func_name: str, platform: str,
    content_id: "str | None" = None,
) -> tuple[str, str, str]:
    """Build the equivalence key identifying a registration.

    Two registrations that share this key produce a functionally identical FFI
    target (same shared-library **content**, same function symbol, same
    platform) and are therefore treated as the *same* registration.  Using
    content rather than the path means a live edit-and-rebuild (same path, new
    bytes) is correctly seen as a *different* registration.

    When the caller can derive a *deterministic* content identity — the
    compilation pipeline passes its cache key, a digest of the source, ABI
    specs, headers, and build options — that is preferred over the ``.so``
    byte hash: compilers embed build paths and timestamps, so recompiling
    *identical* source can yield different bytes, and the byte hash would then
    spuriously refuse a ``force_rebuild`` of unchanged source.
    """
    content = content_id if content_id is not None else _module_content_id(module)
    return (content, str(func_name), str(platform))


def _jax_register(
    target_name: str, module: "CompiledModule", func_name: str, platform: str
) -> None:
    """Perform the raw ``jax.ffi.register_ffi_target`` call for one handler."""
    fn_ptr = module.get_handler(func_name)
    capsule = jax.ffi.pycapsule(fn_ptr)
    jax.ffi.register_ffi_target(target_name, capsule, platform=platform)


def register_ffi_target(
    target_name: str,
    module: CompiledModule,
    func_name: str,
    *,
    platform: str = "CUDA",
    replace: bool = False,
    content_id: "str | None" = None,
) -> None:
    """Register a compiled function as a JAX FFI target.

    After registration, the function can be invoked inside ``@jax.jit``
    via ``jax.ffi.ffi_call(target_name, ...)``.

    The whole check-and-register sequence is guarded by a module-level lock and
    is **idempotent**: re-registering the same ``target_name`` with an
    equivalent module (identical shared-library *content*, function name, and
    platform) is a no-op and does not disturb the live keep-alives.

    A *different* module (e.g. an edited-and-rebuilt kernel) under an
    already-registered name is **refused deterministically on this JAX version,
    regardless of ``replace``** (see Notes): the installed JAX cannot re-point a
    live FFI target to new code in a way that can be performed or verified.  The
    error directs the caller to register the rebuild under a distinct target
    name.  ``replace`` is retained for API stability and reserved for a future
    JAX that supports verifiable re-pointing.

    Parameters
    ----------
    target_name : str
        Globally unique FFI target identifier.
    module : CompiledModule
        The loaded module containing the function.
    func_name : str
        Function name within the module.
    platform : str
        Target platform (``"CUDA"`` or ``"cpu"``).
    replace : bool, default False
        Reserved.  On the installed JAX a content change under an existing
        target name raises whether or not this is set, because a live re-point
        cannot be verified (see Notes).  Kept for API stability / forward
        compatibility; ``force_rebuild=True`` passes it through.
    content_id : str, optional
        Deterministic content identity for the registration, overriding the
        default ``.so`` byte hash.  The compilation pipeline passes its cache
        key (a digest of source, ABI specs, headers, and build options) so
        that recompiling *unchanged* source — whose ``.so`` bytes may still
        differ because compilers embed build paths and timestamps — is
        recognised as the same registration (an idempotent no-op) rather than
        refused, while any real source/spec change still raises.

    Raises
    ------
    KernelRegistrationError
        If ``target_name`` is already registered to *different* ``.so`` content.
        This is deterministic on **all** platforms (CPU and CUDA alike) — see
        Notes for why a live re-point is refused rather than attempted.

    Notes
    -----
    Registration is process-global and intentionally has no unload path: every
    registered ``.so`` is pinned in ``_LIVE_MODULES`` (a list, so future
    re-pointing support can append keep-alives without dropping the old image)
    for the lifetime of the process, so no XLA FFI target ever dangles.

    **Why a live re-point is refused.**  Probed on the installed JAX (0.10.2),
    XLA binds the FFI handler *pointer* into each compiled executable at compile
    time and resolves it by name only once, so already-traced callables keep
    dispatching to the original handler regardless.  Worse, the two platforms
    disagree on re-registration and neither offers a lookup to confirm success:

    - On the CPU/"Host" platform, XLA's registry **rejects** a re-registration
      whose handler pointer differs ("Duplicate FFI handler registration ...
      with different bundle addresses") — a raise.
    - On CUDA, XLA **accepts** the duplicate registration but **silently keeps
      the old handler** — so blindly re-registering would report success while
      the stale kernel keeps executing (this is audit finding 4).

    Because success cannot be positively verified on this JAX, this function
    refuses deterministically instead of guessing.  The reliable way to run
    edited code in a live process is a **distinct target name** (a new
    ``name=`` / ``target_prefix=``), which the error messages recommend; a
    version check can enable true ``replace`` once some JAX supports it.
    """
    key = _registration_key(module, func_name, platform, content_id)

    with _REGISTRATION_LOCK:
        if target_name in _REGISTERED_TARGETS:
            existing = _REGISTRATION_KEYS.get(target_name)
            if existing == key:
                # Equivalent re-registration: no-op, keep the live modules as-is.
                return
            # Different ``.so`` content under an existing target name.  On the
            # installed JAX a live re-point cannot be performed *or verified*
            # (probed): the CPU/Host registry raises on a differing bundle
            # address, the CUDA registry accepts the call but SILENTLY keeps the
            # old handler, and XLA binds the handler pointer into each compiled
            # executable at compile time.  Attempting the re-register would
            # therefore either raise or — worse, on CUDA — report success while
            # the stale kernel keeps executing (audit finding 4).  So we REFUSE
            # deterministically on all platforms, regardless of ``replace``,
            # rather than call _jax_register.  ``replace`` is retained for API
            # stability and gated on a future JAX with verifiable re-pointing.
            if replace:
                raise KernelRegistrationError(
                    f"Cannot replace FFI target '{target_name}': this JAX/XLA "
                    f"cannot re-point a live FFI target to new handler code. "
                    f"Probed on the installed JAX: the CPU/Host registry raises "
                    f"on a differing bundle address; the CUDA registry accepts "
                    f"the call but silently keeps the OLD handler; and XLA binds "
                    f"the handler pointer into each compiled executable at "
                    f"compile time. Register the rebuilt kernel under a distinct "
                    f"name= / target_prefix= instead "
                    f"(existing={existing!r}, requested={key!r})."
                )
            raise KernelRegistrationError(
                f"FFI target '{target_name}' is already registered to different "
                f"content (existing={existing!r}, requested={key!r}). Refusing "
                f"to overwrite the live target. To run edited/rebuilt kernel "
                f"source in a live process, register it under a distinct name= / "
                f"target_prefix= (also required when two sources share a file "
                f"stem). replace=True is reserved for future JAX versions that "
                f"support verifiable re-pointing; on this version it raises too "
                f"— see Notes."
            )

        _jax_register(target_name, module, func_name, platform)

        # Keep the module alive and record its identity.
        _LIVE_MODULES.setdefault(target_name, []).append(module)
        _REGISTERED_TARGETS.add(target_name)
        _REGISTRATION_KEYS[target_name] = key


def list_registered_targets() -> list[str]:
    """Return a sorted list of all registered FFI target names."""
    with _REGISTRATION_LOCK:
        return sorted(_REGISTERED_TARGETS)
