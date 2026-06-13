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

"""Detect and validate the CUDA/C++ compilation toolchain."""

import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import jax

from brainevent._error import (
    KernelToolchainError, NvccNotFoundError, HostCompilerNotFoundError,
    HeaderNotFoundError, GpuArchDetectionError, UnsupportedArchError,
)


# Guards the process-global mutable caches/pins below (``_NVCC_DISCOVERY``,
# ``_COMPUTE_CAPABILITIES``, ``_GPU_DETECT_ERROR``) so concurrent
# ``set_*``/auto-detect callers can't race (audit L12 / Theme G).
_STATE_LOCK = threading.Lock()

# Stashes the most recent exception raised by a GPU-arch detection backend
# (e.g. ``jax.devices("gpu")`` failing on a driver mismatch) so it can be
# surfaced in the ``GpuArchDetectionError`` instead of being flattened into a
# bare "no GPU" (audit M10).  Written under ``_STATE_LOCK``.
_GPU_DETECT_ERROR: "BaseException | None" = None


_ARCH_OK = re.compile(r"^sm_\d{2,3}[a-z]?$")


def normalize_arch(value: "str") -> str:
    """Normalize a compute-capability spec to nvcc's ``sm_XX`` form.

    Accepts ``"8.6"``, ``"86"``, ``"sm_86"``, ``"compute_86"`` and
    architecture-suffixed forms such as ``"9.0a"``/``"90a"``.

    Parameters
    ----------
    value : str
        Compute capability in any common spelling.

    Returns
    -------
    str
        Canonical ``sm_XX`` (optionally suffixed) string.

    Raises
    ------
    ValueError
        If *value* is empty or not syntactically a capability.
    UnsupportedArchError
        If *value* is syntactically valid but names a nonsensical
        architecture with compute-capability major < 2 (e.g. ``"0.0"`` →
        ``sm_00``, ``"1.0"`` → ``sm_10``).  No real CUDA GPU has a major
        version below 2; accepting these only defers the failure to an
        opaque nvcc ``-gencode arch=compute_00`` error much later.
    """
    s = str(value).strip().lower()
    if not s:
        raise ValueError(f"invalid compute capability: {value!r}")
    if s.startswith("sm_"):
        s = s[3:]
    elif s.startswith("compute_"):
        s = s[8:]
    suffix = ""
    if s and s[-1].isalpha():
        suffix, s = s[-1], s[:-1]
    digits = s.replace(".", "")
    if not digits.isdigit() or len(digits) < 2:
        raise ValueError(f"invalid compute capability: {value!r}")
    arch = f"sm_{digits}{suffix}"
    if not _ARCH_OK.match(arch):
        raise ValueError(f"invalid compute capability: {value!r}")
    # Major version is all but the last digit (e.g. sm_86 → 8, sm_120 → 12).
    major = int(digits[:-1])
    if major < 2:
        raise UnsupportedArchError(
            f"unsupported compute capability {value!r} (resolved to {arch}): "
            f"no CUDA GPU has compute-capability major < 2."
        )
    return arch


def _parse_arch_spec(value: "str | list[str]") -> "list[str]":
    """Expand a compute-capability spec into raw, comma-split tokens.

    A single string may carry several capabilities separated by commas
    (e.g. ``"8.6,8.0"``), mirroring the ``BRAINEVENT_COMPUTE_CAPABILITIES``
    env var.  List items are likewise comma-expanded.  Whitespace is trimmed
    and empty tokens dropped.  Tokens are returned verbatim (not normalized);
    callers pass each through :func:`normalize_arch`.
    """
    items = [value] if isinstance(value, str) else list(value)
    tokens: "list[str]" = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens


@dataclass(frozen=True)
class CudaToolchain:
    """Immutable description of available CUDA compilation tools."""
    nvcc: str
    cxx: str
    cuda_home: str
    cuda_include_dirs: tuple[str, ...]
    xla_ffi_include_dir: str
    brainevent_include_dir: str
    nvcc_version: str = ""
    cxx_version: str = ""


@dataclass(frozen=True)
class CppToolchain:
    """Immutable description of the CPU/C++ compilation tools (no CUDA required)."""
    cxx: str
    xla_ffi_include_dir: str
    brainevent_include_dir: str
    cxx_version: str = ""


# ---------------------------------------------------------------------------
# Layered diagnostics: candidate probes + unified error renderer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateProbe:
    """One attempted toolchain candidate and its outcome."""
    source: str   # "BRAINEVENT_NVCC_PATH" / "pip:nvidia/cu13" / "PATH:nvcc" ...
    path: str     # examined path ("" if the env var is unset)
    status: str   # "unset" | "not-found" | "not-a-file" | "rejected:<why>" | "ok"


_PROBE_LABEL = {
    "unset": "unset",
    "not-found": "not found",
    "not-a-file": "not a file",
    "ok": "hit",
}


def _probe_line(p: CandidateProbe) -> str:
    mark = "✓" if p.status == "ok" else "✗"
    if p.status.startswith("rejected:"):
        label = "rejected(" + p.status.split(":", 1)[1] + ")"
    else:
        label = _PROBE_LABEL.get(p.status, p.status)
    loc = f"  [{p.path}]" if p.path else ""
    return f"  {mark} {p.source}{loc}  {label}"


def render_toolchain_error(
    *,
    stage: str,
    code: str,
    summary: str,
    probes: "list[CandidateProbe] | None" = None,
    command: str = "",
    compiler_output: str = "",
    remediation: "list[str] | None" = None,
    snapshot: "dict | None" = None,
) -> str:
    """Render a uniform, layered toolchain/compilation error message."""
    if snapshot is None and os.environ.get("BRAINEVENT_TOOLCHAIN_DEBUG"):
        try:
            snapshot = collect_toolchain_diagnostics()
        except Exception:
            snapshot = None

    lines = [f"[brainevent GPU toolchain] {stage} failed  (code={code})", "", f"Reason: {summary}"]
    if probes:
        lines += ["", "Tried (in priority order):"]
        lines += [_probe_line(p) for p in probes]
    if command:
        lines += ["", "Command:", f"  {command}"]
    if compiler_output:
        lines += ["", "Compiler output:", compiler_output]
    if remediation:
        lines += ["", "How to fix:"]
        lines += [f"  {i}) {r}" for i, r in enumerate(remediation, 1)]
    if snapshot:
        lines += ["", "Toolchain snapshot:"]
        lines += [f"  {k} = {v}" for k, v in snapshot.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# nvcc discovery preference (pip vs system PATH)
# ---------------------------------------------------------------------------

_VALID_NVCC_DISCOVERY = ("pip", "system")
_NVCC_DISCOVERY: "str | None" = None  # None → fall back to env then default


def set_nvcc_discovery(prefer: str) -> None:
    """Set nvcc discovery preference: 'pip' (default) or 'system'.

    The write is guarded by :data:`_STATE_LOCK` so it can't race a concurrent
    setter or auto-detect (audit L12 / Theme G).
    """
    if prefer not in _VALID_NVCC_DISCOVERY:
        raise ValueError(
            f"Invalid nvcc discovery preference {prefer!r}. "
            f"Valid: {_VALID_NVCC_DISCOVERY}"
        )
    global _NVCC_DISCOVERY
    with _STATE_LOCK:
        _NVCC_DISCOVERY = prefer


def get_nvcc_discovery() -> str:
    """Resolve preference: explicit set > BRAINEVENT_NVCC_PREFER env > 'pip'."""
    if _NVCC_DISCOVERY is not None:
        return _NVCC_DISCOVERY
    env = os.environ.get("BRAINEVENT_NVCC_PREFER", "").strip().lower()
    if env in _VALID_NVCC_DISCOVERY:
        return env
    return "pip"


# ---------------------------------------------------------------------------
# Compute-capability pin (process-global override of auto-detection)
# ---------------------------------------------------------------------------

_COMPUTE_CAPABILITIES: "list[str] | None" = None


def set_compute_capabilities(value: "str | list[str] | None") -> None:
    """Pin the target compute capability/-ies process-wide (``None`` = auto).

    The write is guarded by :data:`_STATE_LOCK` so it can't race a concurrent
    setter or auto-detect (audit L12 / Theme G).  Normalization (which may
    raise :class:`ValueError`/:class:`~brainevent._error.UnsupportedArchError`)
    happens before the lock is taken so a bad value never mutates global state.
    """
    global _COMPUTE_CAPABILITIES
    if value is None:
        with _STATE_LOCK:
            _COMPUTE_CAPABILITIES = None
        return
    tokens = _parse_arch_spec(value)
    normalized = _dedup([normalize_arch(v) for v in tokens]) if tokens else None
    with _STATE_LOCK:
        _COMPUTE_CAPABILITIES = normalized


def get_compute_capabilities() -> "list[str] | None":
    """Return the pinned compute capabilities, or ``None`` if auto."""
    return list(_COMPUTE_CAPABILITIES) if _COMPUTE_CAPABILITIES is not None else None


# ---------------------------------------------------------------------------
# Discovery helpers: pip-installed nvcc, host compiler, selection
# ---------------------------------------------------------------------------

def _nvcc_name() -> str:
    return "nvcc.exe" if sys.platform == "win32" else "nvcc"


def _nvidia_roots() -> "list[str]":
    """Return search roots of the 'nvidia' namespace package (empty if absent)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return list(spec.submodule_search_locations)


def _pip_include_for(pkg: str, roots: "list[str]") -> "str | None":
    for root in roots:
        inc = Path(root) / pkg / "include"
        if inc.is_dir():
            return str(inc)
    return None


def _find_pip_cuda(roots: "list[str] | None" = None):
    """Locate a pip-installed CUDA nvcc + its include dirs.

    Returns ``((nvcc_path, [include_dirs]) | None, [CandidateProbe])``.
    Handles consolidated ``nvidia/cuNN`` (cu13+) and split ``nvidia/cuda_nvcc``
    (cu12) layouts; consolidated wins, highest cuNN wins.
    """
    if roots is None:
        roots = _nvidia_roots()
    probes: "list[CandidateProbe]" = []
    exe = _nvcc_name()

    # A. consolidated cuNN layout (cu13 and future cuNN)
    best = None  # (version:int, nvcc_path:str, include_dir:str)
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for entry in sorted(rp.iterdir()):
            if not re.match(r"cu\d+$", entry.name):
                continue
            cand = entry / "bin" / exe
            src = f"pip:nvidia/{entry.name}"
            if cand.is_file():
                ver = int(entry.name[2:])
                probes.append(CandidateProbe(src, str(cand), "ok"))
                if best is None or ver > best[0]:
                    best = (ver, str(cand), str(entry / "include"))
            else:
                probes.append(CandidateProbe(src, str(cand), "not-found"))
    if best is not None:
        return (best[1], [best[2]]), probes

    # B. split cu12 layout
    for root in roots:
        cand = Path(root) / "cuda_nvcc" / "bin" / exe
        src = "pip:nvidia/cuda_nvcc"
        if cand.is_file():
            includes = [str(Path(root) / "cuda_nvcc" / "include")]
            for pkg in ("cuda_runtime", "cuda_cccl"):
                inc = _pip_include_for(pkg, roots)
                if inc:
                    includes.append(inc)
            probes.append(CandidateProbe(src, str(cand), "ok"))
            return (str(cand), includes), probes
        probes.append(CandidateProbe(src, str(cand), "not-found"))

    if not roots:
        probes.append(CandidateProbe("pip:nvidia/*", "", "not-found"))
    return None, probes


def _find_host_cxx():
    """Find a host C++ compiler: CXX > conda > system PATH.

    Returns ``(cxx_path | None, [CandidateProbe])``.
    """
    probes: "list[CandidateProbe]" = []

    # 1. CXX
    cxx_env = os.environ.get("CXX")
    if cxx_env:
        resolved = cxx_env if os.path.isfile(cxx_env) else shutil.which(cxx_env)
        if resolved:
            probes.append(CandidateProbe("CXX", resolved, "ok"))
            return resolved, probes
        probes.append(CandidateProbe("CXX", cxx_env, "not-found"))
    else:
        probes.append(CandidateProbe("CXX", "", "unset"))

    # 2. conda
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        bindir = Path(conda) / "bin"
        names = [p.name for p in sorted(bindir.glob("*-g++"))] if bindir.is_dir() else []
        names += ["g++", "c++", "clang++"]
        for name in names:
            cand = bindir / name
            src = f"$CONDA_PREFIX/bin/{name}"
            if cand.is_file():
                probes.append(CandidateProbe(src, str(cand), "ok"))
                return str(cand), probes
            probes.append(CandidateProbe(src, str(cand), "not-found"))
    else:
        probes.append(CandidateProbe("$CONDA_PREFIX", "", "unset"))

    # 2b. Windows: MSVC cl.exe (nvcc's required host compiler there) — preferred
    #     over g++/clang on Windows, so probe it before the generic PATH search.
    if sys.platform == "win32":
        cl = shutil.which("cl") or shutil.which("cl.exe")
        if cl:
            probes.append(CandidateProbe("PATH:cl", cl, "ok"))
            return cl, probes
        probes.append(CandidateProbe("PATH:cl", "", "not-found"))
        vswhere = os.path.join(
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
            "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(vswhere):
            try:
                out = subprocess.run(
                    [vswhere, "-latest", "-products", "*", "-find",
                     "VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64\\cl.exe"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                if out:
                    probes.append(CandidateProbe("vswhere:cl", out[0], "ok"))
                    return out[0], probes
            except (OSError, subprocess.SubprocessError):
                pass
            probes.append(CandidateProbe("vswhere:cl", vswhere, "not-found"))

    # 3. system PATH
    for name in ("g++", "c++", "clang++"):
        found = shutil.which(name)
        if found:
            probes.append(CandidateProbe(f"PATH:{name}", found, "ok"))
            return found, probes
        probes.append(CandidateProbe(f"PATH:{name}", "", "not-found"))

    return None, probes


def _include_from_nvcc(nvcc_path: str) -> str:
    """Guess the CUDA include dir as ``<nvcc>/../../include``.

    This ``parent.parent`` heuristic holds for a real CUDA Toolkit layout
    (``<prefix>/bin/nvcc`` ↔ ``<prefix>/include``) but can be wrong for a
    distro shim at ``/usr/bin/nvcc`` (whose headers live in
    ``/usr/include`` or a versioned ``/usr/lib/cuda/include``).  Callers
    that depend on the result validate it via :func:`_validate_cuda_include`
    (audit L12).
    """
    return str(Path(nvcc_path).resolve().parent.parent / "include")


def _validate_cuda_include(
    include_dirs: "list[str]", *, nvcc_probes: "list[CandidateProbe]"
) -> None:
    """Verify that ``cuda_runtime.h`` is reachable from *include_dirs*.

    Parameters
    ----------
    include_dirs : list of str
        Candidate CUDA include directories (as resolved alongside nvcc).
    nvcc_probes : list of CandidateProbe
        The discovery probes for the chosen nvcc, echoed into the error for
        context.

    Raises
    ------
    HeaderNotFoundError
        If none of *include_dirs* contains ``cuda_runtime.h``.  This catches
        the distro-``/usr/bin/nvcc``-shim failure mode where the
        ``<nvcc>/../../include`` guess points at a directory with no CUDA
        headers (audit L12), instead of letting compilation fail later with
        an opaque ``cuda_runtime.h: No such file or directory``.
    """
    header_probes: "list[CandidateProbe]" = []
    for d in include_dirs:
        cand = os.path.join(d, "cuda_runtime.h")
        if os.path.isfile(cand):
            return
        header_probes.append(CandidateProbe(
            "cuda_runtime.h", cand,
            "not-found" if os.path.isdir(d) else "not-a-file",
        ))
    if not include_dirs:
        header_probes.append(CandidateProbe("cuda_runtime.h", "", "unset"))
    raise HeaderNotFoundError(render_toolchain_error(
        stage="header resolution", code="E-HDR",
        summary=(
            "CUDA header cuda_runtime.h was not found in the include "
            "directory resolved from nvcc. The nvcc may be a distro shim "
            "whose headers live elsewhere, or the CUDA install is incomplete."
        ),
        probes=(nvcc_probes or []) + header_probes,
        remediation=[
            "Install the matching CUDA headers (e.g. the cuda-cudart-dev / "
            "cuda-runtime package), or",
            "Set CUDA_HOME (or CUDA_PATH) to a complete CUDA Toolkit install, or",
            'Install a pip CUDA wheel that bundles headers: pip install -U "jax[cuda13]"',
        ],
    ))


def _cxx_version(cxx: str) -> str:
    """Return the first line of *cxx*'s ``--version`` banner (``""`` on failure).

    Parameters
    ----------
    cxx : str
        Path to (or name of) the host C++ compiler.

    Returns
    -------
    str
        The compiler's version banner's first line, or ``""`` if the compiler
        is missing/unrunnable.

    Notes
    -----
    The empty-string degrade is deliberately narrow (audit M8):

    * Only ``OSError`` (missing/permission) and
      :class:`subprocess.SubprocessError` (timeouts, etc.) are swallowed --
      a ``KeyboardInterrupt`` or other ``BaseException`` propagates.
    * The banner is read from ``stdout`` **or** ``stderr``: MSVC's ``cl.exe``
      prints its version to stderr, so reading only stdout silently yielded an
      empty version that then collided in the compilation cache key.
    * Decoding uses ``errors="replace"`` so a non-UTF-8 byte in the banner
      degrades to a readable version string rather than raising.
    """
    try:
        proc = subprocess.run(
            [cxx, "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    out = proc.stdout or proc.stderr or ""
    lines = out.splitlines()
    return lines[0].strip() if lines else ""


def _select_nvcc():
    """Pick nvcc per priority. Returns ``(nvcc|None, [include_dirs], [probes])``."""
    probes: "list[CandidateProbe]" = []
    exe = _nvcc_name()

    # 1. BRAINEVENT_NVCC_PATH
    env = os.environ.get("BRAINEVENT_NVCC_PATH")
    if env:
        if os.path.isfile(env):
            probes.append(CandidateProbe("BRAINEVENT_NVCC_PATH", env, "ok"))
            return env, [_include_from_nvcc(env)], probes
        probes.append(CandidateProbe("BRAINEVENT_NVCC_PATH", env, "not-a-file"))
    else:
        probes.append(CandidateProbe("BRAINEVENT_NVCC_PATH", "", "unset"))

    # 2. explicit CUDA root env vars. ``CUDA_HOME`` is the conventional Linux
    #    name; the Windows CUDA installer instead sets ``CUDA_PATH`` (never
    #    ``CUDA_HOME``), so both must be probed or standard Windows installs go
    #    undiscovered when nvcc is off PATH (audit H6).
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        src = f"${var}/bin/nvcc"
        if home:
            cand = os.path.join(home, "bin", exe)
            if os.path.isfile(cand):
                probes.append(CandidateProbe(src, cand, "ok"))
                return cand, [os.path.join(home, "include")], probes
            probes.append(CandidateProbe(src, cand, "not-found"))
        else:
            probes.append(CandidateProbe(src, "", "unset"))

    # 3. preference-ordered pip vs system PATH
    order = ("pip", "system") if get_nvcc_discovery() == "pip" else ("system", "pip")
    for src in order:
        if src == "pip":
            res, pip_probes = _find_pip_cuda()
            probes.extend(pip_probes)
            if res is not None:
                return res[0], res[1], probes
        else:
            which = shutil.which("nvcc")
            if which:
                probes.append(CandidateProbe("PATH:nvcc", which, "ok"))
                return which, [_include_from_nvcc(which)], probes
            probes.append(CandidateProbe("PATH:nvcc", "", "not-found"))

    return None, [], probes


def collect_toolchain_diagnostics() -> dict:
    """Single source of truth for a toolchain snapshot (never raises)."""
    snap: "dict[str, str]" = {}
    snap["discovery"] = get_nvcc_discovery()
    try:
        nvcc, includes, _ = _select_nvcc()
    except Exception:
        nvcc, includes = None, []
    snap["nvcc"] = nvcc or "<not found>"
    snap["cuda_include_dirs"] = ", ".join(includes) if includes else "<none>"
    try:
        cxx, _ = _find_host_cxx()
    except Exception:
        cxx = None
    snap["host_cxx"] = cxx or "<not found>"
    if cxx:
        snap["host_cxx_version"] = _cxx_version(cxx)
    # Surface a stashed GPU-arch detection failure (e.g. a CUDA driver
    # mismatch raised by ``jax.devices("gpu")``) so it shows up in debug
    # snapshots, not just in the GpuArchDetectionError (audit M10).
    with _STATE_LOCK:
        detect_err = _GPU_DETECT_ERROR
    if detect_err is not None:
        snap["gpu_detect_error"] = f"{type(detect_err).__name__}: {detect_err}"
    for var in ("BRAINEVENT_NVCC_PATH", "CUDA_HOME", "CUDA_PATH", "CXX",
                "CONDA_PREFIX", "BRAINEVENT_NVCC_PREFER",
                "BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER",
                "BRAINEVENT_COMPUTE_CAPABILITIES"):
        snap[f"env:{var}"] = os.environ.get(var, "<unset>")
    return snap


def _resolve_xla_ffi_include(summary: str, remediation: "list[str]") -> str:
    """Return the jaxlib XLA FFI include dir, or raise if ``ffi.h`` is missing."""
    xla_ffi_include = jax.ffi.include_dir()
    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="header resolution", code="E-HDR",
            summary=summary,
            probes=[CandidateProbe("jaxlib:xla/ffi/api/ffi.h", ffi_header, "not-found")],
            remediation=remediation,
        ))
    return xla_ffi_include


def _resolve_brainevent_include() -> str:
    """Return the brainevent include dir, or raise if it is missing."""
    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="header resolution", code="E-HDR",
            summary="brainevent include directory is missing (the package install may be corrupted).",
            probes=[CandidateProbe("brainevent/include", be_include, "not-found")],
            remediation=["Reinstall brainevent: pip install -U --force-reinstall brainevent"],
        ))
    return be_include


def detect_cuda_toolchain() -> CudaToolchain:
    """Auto-detect nvcc, host C++ compiler, and include paths.

    Raises the matching layered exception (NvccNotFoundError /
    HostCompilerNotFoundError / HeaderNotFoundError) when a stage fails.
    """
    nvcc, cuda_include_dirs, nvcc_probes = _select_nvcc()
    if nvcc is None:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc discovery", code="E-NVCC",
            summary="CUDA compiler nvcc not found.",
            probes=nvcc_probes,
            remediation=[
                'Install jax[cuda13] (bundles nvcc, no system CUDA Toolkit required): pip install -U "jax[cuda13]"',
                "Or set BRAINEVENT_NVCC_PATH to the nvcc path, or set CUDA_HOME to the CUDA install directory",
                "To prefer the nvcc on the system PATH: brainevent.config.prefer_system_nvcc()",
            ],
        ))

    cuda_home = str(Path(nvcc).resolve().parent.parent)

    # nvcc version
    nvcc_version = ""
    try:
        proc = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc discovery", code="E-NVCC",
            summary=f"nvcc could not be executed: {nvcc} ({e.__class__.__name__})",
            probes=nvcc_probes,
            remediation=["Verify this nvcc is executable and not corrupted, or use a different nvcc."],
        )) from e
    # nvcc *ran* but exited non-zero (loader/GLIBC/driver mismatch, a stub
    # wrapper, ...): stdout is typically empty so the "release" scan below
    # would find nothing and misreport this as "version could not be
    # determined". Surface the real status + captured output instead (audit
    # H5), mirroring the returncode check in ``_arch_from_nvidia_smi``.
    if proc.returncode != 0:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc discovery", code="E-NVCC",
            summary=(
                f"nvcc exited with status {proc.returncode}: it was found and "
                f"launched but failed to run: {nvcc}"
            ),
            probes=nvcc_probes,
            compiler_output=((proc.stdout or "") + (proc.stderr or "")),
            remediation=[
                "Check that nvcc's shared-library dependencies resolve (e.g. "
                "ldd nvcc), the CUDA driver matches the toolkit, and it is not "
                "a stub wrapper; or use a different nvcc.",
            ],
        ))
    for line in proc.stdout.splitlines():
        if "release" in line.lower():
            nvcc_version = line.strip()
            break
    if not nvcc_version:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc discovery", code="E-NVCC",
            summary=f"nvcc exists but its version could not be determined: {nvcc}",
            probes=nvcc_probes,
            compiler_output=((proc.stdout or "") + (proc.stderr or "")),
            remediation=["Verify this nvcc is executable and not corrupted, or use a different nvcc."],
        ))

    # Validate the resolved CUDA include dir actually has cuda_runtime.h
    # before we hand the toolchain to the compiler (audit L12): a distro
    # /usr/bin/nvcc shim's ``../../include`` guess can point at a dir with no
    # CUDA headers, which would otherwise fail much later with an opaque
    # "cuda_runtime.h: No such file or directory".
    _validate_cuda_include(list(cuda_include_dirs), nvcc_probes=nvcc_probes)

    # host compiler
    cxx, cxx_probes = _find_host_cxx()
    if cxx is None:
        raise HostCompilerNotFoundError(render_toolchain_error(
            stage="host compiler discovery", code="E-CXX",
            summary="host C++ compiler not found (nvcc needs it to compile and link host-side code). pip does not provide a host compiler.",
            probes=cxx_probes,
            remediation=[
                "conda environment: conda install -c conda-forge gxx",
                "Debian/Ubuntu: sudo apt-get install g++",
                "RHEL/Fedora: sudo dnf install gcc-c++",
                "Or set CXX=/path/to/g++",
            ],
        ))
    cxx_version = _cxx_version(cxx)

    xla_ffi_include = _resolve_xla_ffi_include(
        summary="XLA FFI header ffi.h not found (jaxlib and the CUDA wheel may be mismatched or corrupted).",
        remediation=['Reinstall a jaxlib matching your CUDA: pip install -U "jax[cuda13]"'],
    )
    be_include = _resolve_brainevent_include()

    return CudaToolchain(
        nvcc=nvcc,
        cxx=cxx,
        cuda_home=cuda_home,
        cuda_include_dirs=tuple(cuda_include_dirs),
        xla_ffi_include_dir=xla_ffi_include,
        brainevent_include_dir=be_include,
        nvcc_version=nvcc_version,
        cxx_version=cxx_version,
    )


def detect_cpp_toolchain() -> CppToolchain:
    """Auto-detect a host C++ compiler and include paths for CPU-only compilation.

    Does not require CUDA to be installed.
    """
    cxx, cxx_probes = _find_host_cxx()
    if cxx is None:
        raise HostCompilerNotFoundError(render_toolchain_error(
            stage="host compiler discovery", code="E-CXX",
            summary="C++ compiler not found (the CPU backend requires g++/clang++).",
            probes=cxx_probes,
            remediation=[
                "conda environment: conda install -c conda-forge gxx",
                "Debian/Ubuntu: sudo apt-get install g++",
                "RHEL/Fedora: sudo dnf install gcc-c++",
                "Or set CXX=/path/to/g++",
            ],
        ))
    cxx_version = _cxx_version(cxx)

    xla_ffi_include = _resolve_xla_ffi_include(
        summary="XLA FFI header ffi.h not found.",
        remediation=["Reinstall jaxlib: pip install -U jax jaxlib"],
    )
    be_include = _resolve_brainevent_include()

    return CppToolchain(
        cxx=cxx,
        xla_ffi_include_dir=xla_ffi_include,
        brainevent_include_dir=be_include,
        cxx_version=cxx_version,
    )


# ---------------------------------------------------------------------------
# Cross-platform helpers
# ---------------------------------------------------------------------------

def so_ext() -> str:
    """Return the shared-library file extension for the current OS.

    +----------+-----------+
    | Platform | Extension |
    +==========+===========+
    | Linux    | ``.so``   |
    | macOS    | ``.dylib``|
    | Windows  | ``.dll``  |
    +----------+-----------+
    """
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    if sys.platform.startswith("linux"):
        return ".so"
    raise KernelToolchainError('Unsupported platform: %s' % sys.platform)


def _is_msvc(cxx: str) -> bool:
    """Return True if *cxx* is the MSVC compiler (``cl.exe``)."""
    return os.path.basename(cxx).lower() in ("cl", "cl.exe")


def cxx_shared_flags(cxx: str) -> list[str]:
    """Return C++ compiler flags to build a shared library.

    Handles GCC/Clang on Linux/macOS, MSVC on Windows, and MinGW on Windows.

    Parameters
    ----------
    cxx : str
        Path to the C++ compiler executable.
    """
    if sys.platform == "darwin":
        return ["-dynamiclib", "-fPIC"]
    if sys.platform == "win32":
        if _is_msvc(cxx):
            return ["/LD", "/MD"]
        return ["-shared"]  # MinGW — PE format, no -fPIC needed
    if sys.platform.startswith("linux"):
        return ["-shared", "-fPIC"]  # Linux
    raise KernelToolchainError('Unsupported platform: %s' % sys.platform)


def cxx_std_flag(cxx: str) -> str:
    """Return the C++17 standard flag for *cxx*.

    MSVC uses ``/std:c++17``; all other compilers use ``-std=c++17``.
    """
    return "/std:c++17" if _is_msvc(cxx) else "-std=c++17"


def nvcc_host_pic_flags() -> list[str]:
    """Return nvcc flags for position-independent code on the host.

    On Linux/macOS, nvcc needs ``--compiler-options -fPIC`` to pass the flag
    to the host compiler.  On Windows (PE format), no equivalent is needed.
    """
    if sys.platform == "win32":
        return []
    return ["--compiler-options", "-fPIC"]


def _dedup(seq: "list[str]") -> "list[str]":
    """Return *seq* with duplicates removed, order preserved."""
    out: "list[str]" = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _arch_from_jax() -> "list[str] | None":
    """Compute capabilities of visible JAX GPU devices (never raises).

    Returns device order (device 0 first), so the first element is JAX's
    default device.  ``None`` when no GPU device / attribute is available.

    A *broken* CUDA backend (e.g. ``jax.devices("gpu")`` raising on a
    driver/runtime mismatch) is distinct from "no GPU present": the former's
    exception is stashed in the module-global ``_GPU_DETECT_ERROR`` so
    :func:`resolve_compute_capabilities` can surface the real cause instead of
    a misleading "no GPU detected" (audit M10).
    """
    try:
        devices = jax.devices("gpu")
    except Exception as e:  # noqa: BLE001 - intentionally broad: any backend fault
        global _GPU_DETECT_ERROR
        with _STATE_LOCK:
            _GPU_DETECT_ERROR = e
        return None
    caps: "list[str]" = []
    for d in devices:
        cc = getattr(d, "compute_capability", None)
        if not cc:
            continue
        try:
            caps.append(normalize_arch(cc))
        except (ValueError, UnsupportedArchError):
            continue
    return _dedup(caps) or None


def _arch_from_nvidia_smi() -> "list[str] | None":
    """Compute capabilities via the nvidia-smi binary (never raises)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    caps: "list[str]" = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            caps.append(normalize_arch(line))
        except (ValueError, UnsupportedArchError):
            continue
    return _dedup(caps) or None


def resolve_compute_capabilities(explicit: "str | list[str] | None" = None) -> "list[str]":
    """Resolve target compute capabilities by precedence.

    Order (first hit wins): *explicit* argument >
    :func:`set_compute_capabilities` pin > the
    ``BRAINEVENT_COMPUTE_CAPABILITIES`` env var > JAX device query >
    nvidia-smi > raise :class:`GpuArchDetectionError`.

    Every source is passed through :func:`normalize_arch`.
    """
    if explicit:
        tokens = _parse_arch_spec(explicit)
        if tokens:
            return _dedup([normalize_arch(v) for v in tokens])
    if _COMPUTE_CAPABILITIES is not None:
        return list(_COMPUTE_CAPABILITIES)
    env = os.environ.get("BRAINEVENT_COMPUTE_CAPABILITIES", "").strip()
    if env:
        out = _dedup([normalize_arch(v) for v in _parse_arch_spec(env)])
        if out:
            return out
    # Clear any stale detection error before re-probing the live backends.
    global _GPU_DETECT_ERROR
    with _STATE_LOCK:
        _GPU_DETECT_ERROR = None
    for source in (_arch_from_jax, _arch_from_nvidia_smi):
        arches = source()
        if arches:
            return arches

    # A backend that *raised* (e.g. JAX's CUDA backend failing on a
    # driver/runtime mismatch) is reported distinctly from "no GPU present",
    # so the operator isn't sent down the wrong remediation path (audit M10).
    with _STATE_LOCK:
        cause = _GPU_DETECT_ERROR
    if cause is not None:
        summary = (
            "GPU compute-capability detection failed: a CUDA backend raised "
            f"while querying devices ({type(cause).__name__}). This usually "
            "means a broken/mismatched CUDA driver or runtime, not the absence "
            "of a GPU."
        )
        compiler_output = f"{type(cause).__name__}: {cause}"
    else:
        summary = ("Could not detect a GPU compute capability "
                   "(no JAX GPU device, and nvidia-smi unavailable).")
        compiler_output = ""
    raise GpuArchDetectionError(render_toolchain_error(
        stage="compute capability detection", code="E-ARCH",
        summary=summary,
        probes=[CandidateProbe("nvidia-smi", shutil.which("nvidia-smi") or "", "not-found")],
        compiler_output=compiler_output,
        remediation=[
            "Run on a machine with a visible GPU, or",
            "Set BRAINEVENT_COMPUTE_CAPABILITIES (e.g. '8.6,8.0'), or",
            "Call brainevent.config.set_compute_capability('8.6'), or",
            "Pass compute_capability='sm_86' to the load function",
        ],
    )) from cause


def detect_cuda_arch() -> "list[str]":
    """Auto-detect GPU compute capabilities.

    Resolves via :func:`resolve_compute_capabilities` (config/env pins, then
    JAX device query, then nvidia-smi).  Raises :class:`GpuArchDetectionError`
    when none are available.
    """
    return resolve_compute_capabilities()


def gencode_flags(arches: "list[str]") -> "list[str]":
    """Return nvcc ``-gencode`` flags: native SASS per arch + PTX for the highest.

    The PTX (``code=compute_X``) for the newest arch keeps the binary
    forward-compatible: it JITs onto GPU generations newer than any compiled
    SASS target.

    Parameters
    ----------
    arches : list of str
        Target architectures in any spelling accepted by :func:`normalize_arch`.

    Returns
    -------
    list of str
        Flattened ``-gencode`` argument list.

    Raises
    ------
    ValueError
        If *arches* is empty.
    """
    if not arches:
        raise ValueError("gencode_flags requires at least one architecture")
    norm = _dedup([normalize_arch(a) for a in arches])
    flags: "list[str]" = []
    for a in norm:
        num = a[3:]
        flags += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    highest = max(norm, key=lambda a: int("".join(c for c in a[3:] if c.isdigit())))
    hnum = highest[3:]
    flags += ["-gencode", f"arch=compute_{hnum},code=compute_{hnum}"]
    return flags
