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
import re
import threading

import jax

from brainevent._error import KernelError, KernelLoadError, KernelRegistrationError


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
        try:
            self._lib = ctypes.CDLL(self._so_path)
        except OSError as e:
            raise KernelLoadError(_format_load_error(self._so_path, e)) from e
        self._functions: dict[str, ctypes._CFuncPtr] = {}

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
    def function_names(self) -> list[str]:
        """Names of available functions."""
        return list(self._functions)

    def __repr__(self) -> str:
        return f"CompiledModule(path={self._so_path!r}, functions={self.function_names})"


# ---------------------------------------------------------------------------
# JAX FFI registration bridge
# ---------------------------------------------------------------------------

# Global registry of (target_name → CompiledModule) to prevent garbage
# collection of the ctypes CDLL while the FFI target is alive.
_LIVE_MODULES: dict[str, CompiledModule] = {}

# Track registered names to give clear errors on duplicates.
_REGISTERED_TARGETS: set[str] = set()

# Identity of each registration, used to decide whether a same-name
# re-registration is an idempotent no-op (equivalent module) or a conflicting
# clobber (different module).  Maps target_name → (so_path, func_name, platform).
_REGISTRATION_KEYS: dict[str, tuple[str, str, str]] = {}

# Serialises the check-and-register sequence below.  ``jax.ffi.register_ffi_target``
# *silently overwrites* an existing target, so without this lock two threads can
# both pass the membership check and double-register (one of them clobbering the
# other's live module, dropping a still-referenced keep-alive).  Guarding the
# whole read-modify-write of ``_REGISTERED_TARGETS`` / ``_LIVE_MODULES`` /
# ``_REGISTRATION_KEYS`` makes registration atomic.
_REGISTRATION_LOCK = threading.Lock()


def _registration_key(
    module: "CompiledModule", func_name: str, platform: str
) -> tuple[str, str, str]:
    """Build the equivalence key identifying a registration.

    Two registrations that share this key produce a functionally identical FFI
    target (same shared-library path, same function symbol, same platform) and
    are therefore treated as the *same* registration.
    """
    return (str(getattr(module, "path", module)), str(func_name), str(platform))


def register_ffi_target(
    target_name: str,
    module: CompiledModule,
    func_name: str,
    *,
    platform: str = "CUDA",
) -> None:
    """Register a compiled function as a JAX FFI target.

    After registration, the function can be invoked inside ``@jax.jit``
    via ``jax.ffi.ffi_call(target_name, ...)``.

    The whole check-and-register sequence is guarded by a module-level lock and
    is **idempotent**: re-registering the same ``target_name`` with an
    equivalent module (identical shared-library path, function name, and
    platform) is a no-op and does not overwrite the live keep-alive.  Attempting
    to register a *different* module under an already-registered name raises
    :class:`~brainevent._error.KernelRegistrationError` rather than silently
    clobbering the previous target (``jax.ffi.register_ffi_target`` would
    otherwise overwrite it without warning, dropping a still-referenced module).

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

    Raises
    ------
    KernelRegistrationError
        If ``target_name`` is already registered to a *different* module,
        function, or platform.

    Notes
    -----
    Registration is process-global and intentionally has no unload path: the
    ctypes ``CDLL`` (and therefore the loaded ``.so``) is pinned in
    ``_LIVE_MODULES`` for the lifetime of the process so the XLA FFI target it
    backs never dangles.  The idempotency rule above bounds this to one live
    module per target name (rather than leaking a new one per duplicate call).
    """
    key = _registration_key(module, func_name, platform)

    with _REGISTRATION_LOCK:
        if target_name in _REGISTERED_TARGETS:
            existing = _REGISTRATION_KEYS.get(target_name)
            if existing == key:
                # Equivalent re-registration: no-op, keep the live module as-is.
                return
            raise KernelRegistrationError(
                f"FFI target '{target_name}' is already registered to a different "
                f"module (existing={existing!r}, requested={key!r}). Refusing to "
                f"overwrite the live target; use a distinct target name."
            )

        fn_ptr = module.get_handler(func_name)
        capsule = jax.ffi.pycapsule(fn_ptr)
        jax.ffi.register_ffi_target(target_name, capsule, platform=platform)

        # Keep the module alive and record its identity.
        _LIVE_MODULES[target_name] = module
        _REGISTERED_TARGETS.add(target_name)
        _REGISTRATION_KEYS[target_name] = key


def list_registered_targets() -> list[str]:
    """Return a sorted list of all registered FFI target names."""
    with _REGISTRATION_LOCK:
        return sorted(_REGISTERED_TARGETS)
