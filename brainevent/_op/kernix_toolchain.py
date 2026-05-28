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
from dataclasses import dataclass
from pathlib import Path

import jax

from brainevent._error import (
    KernelToolchainError, NvccNotFoundError, HostCompilerNotFoundError,
    HeaderNotFoundError, GpuArchDetectionError,
)


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
    "unset": "未设置",
    "not-found": "未找到",
    "not-a-file": "不是文件",
    "ok": "命中",
}


def _probe_line(p: CandidateProbe) -> str:
    mark = "✓" if p.status == "ok" else "✗"
    if p.status.startswith("rejected:"):
        label = "拒绝(" + p.status.split(":", 1)[1] + ")"
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

    lines = [f"[brainevent GPU 工具链] {stage} 失败  (code={code})", "", f"原因: {summary}"]
    if probes:
        lines += ["", "已尝试 (按优先级):"]
        lines += [_probe_line(p) for p in probes]
    if command:
        lines += ["", "命令:", f"  {command}"]
    if compiler_output:
        lines += ["", "编译器输出:", compiler_output]
    if remediation:
        lines += ["", "如何修复:"]
        lines += [f"  {i}) {r}" for i, r in enumerate(remediation, 1)]
    if snapshot:
        lines += ["", "工具链快照:"]
        lines += [f"  {k} = {v}" for k, v in snapshot.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# nvcc discovery preference (pip vs system PATH)
# ---------------------------------------------------------------------------

_VALID_NVCC_DISCOVERY = ("pip", "system")
_NVCC_DISCOVERY: "str | None" = None  # None → fall back to env then default


def set_nvcc_discovery(prefer: str) -> None:
    """Set nvcc discovery preference: 'pip' (default) or 'system'."""
    if prefer not in _VALID_NVCC_DISCOVERY:
        raise ValueError(
            f"Invalid nvcc discovery preference {prefer!r}. "
            f"Valid: {_VALID_NVCC_DISCOVERY}"
        )
    global _NVCC_DISCOVERY
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

    # 3. system PATH
    for name in ("g++", "c++", "clang++"):
        found = shutil.which(name)
        if found:
            probes.append(CandidateProbe(f"PATH:{name}", found, "ok"))
            return found, probes
        probes.append(CandidateProbe(f"PATH:{name}", "", "not-found"))

    return None, probes


def _include_from_nvcc(nvcc_path: str) -> str:
    return str(Path(nvcc_path).resolve().parent.parent / "include")


def _cxx_version(cxx: str) -> str:
    try:
        proc = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=10)
        return proc.stdout.splitlines()[0].strip() if proc.stdout else ""
    except Exception:
        return ""


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

    # 2. explicit CUDA_HOME
    home = os.environ.get("CUDA_HOME")
    if home:
        cand = os.path.join(home, "bin", exe)
        if os.path.isfile(cand):
            probes.append(CandidateProbe("$CUDA_HOME/bin/nvcc", cand, "ok"))
            return cand, [os.path.join(home, "include")], probes
        probes.append(CandidateProbe("$CUDA_HOME/bin/nvcc", cand, "not-found"))
    else:
        probes.append(CandidateProbe("$CUDA_HOME/bin/nvcc", "", "unset"))

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
    for var in ("BRAINEVENT_NVCC_PATH", "CUDA_HOME", "CXX", "CONDA_PREFIX",
                "BRAINEVENT_NVCC_PREFER", "BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER",
                "BRAINEVENT_COMPUTE_CAPABILITIES"):
        snap[f"env:{var}"] = os.environ.get(var, "<unset>")
    return snap


def detect_cuda_toolchain() -> CudaToolchain:
    """Auto-detect nvcc, host C++ compiler, and include paths.

    Raises the matching layered exception (NvccNotFoundError /
    HostCompilerNotFoundError / HeaderNotFoundError) when a stage fails.
    """
    nvcc, cuda_include_dirs, nvcc_probes = _select_nvcc()
    if nvcc is None:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc 发现", code="E-NVCC",
            summary="未找到 CUDA 编译器 nvcc。",
            probes=nvcc_probes,
            remediation=[
                '安装 jax[cuda13]（自带 nvcc，无需系统 CUDA Toolkit）：pip install -U "jax[cuda13]"',
                "或设 BRAINEVENT_NVCC_PATH 指向 nvcc，或设 CUDA_HOME 指向 CUDA 安装目录",
                "若想优先用系统 PATH 上的 nvcc：brainevent.config.prefer_system_nvcc()",
            ],
        ))

    cuda_home = str(Path(nvcc).resolve().parent.parent)

    # nvcc version
    nvcc_version = ""
    proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
    for line in proc.stdout.splitlines():
        if "release" in line.lower():
            nvcc_version = line.strip()
            break
    if not nvcc_version:
        raise NvccNotFoundError(render_toolchain_error(
            stage="nvcc 发现", code="E-NVCC",
            summary=f"nvcc 存在但无法获取版本：{nvcc}",
            probes=nvcc_probes,
            compiler_output=(proc.stdout + proc.stderr),
            remediation=["确认该 nvcc 可执行且未损坏，或改用其它 nvcc。"],
        ))

    # host compiler
    cxx, cxx_probes = _find_host_cxx()
    if cxx is None:
        raise HostCompilerNotFoundError(render_toolchain_error(
            stage="host 编译器发现", code="E-CXX",
            summary="未找到 host C++ 编译器（nvcc 需要它编译 host 侧代码并链接）。pip 不提供 host 编译器。",
            probes=cxx_probes,
            remediation=[
                "conda 环境：conda install -c conda-forge gxx",
                "Debian/Ubuntu：sudo apt-get install g++",
                "RHEL/Fedora：sudo dnf install gcc-c++",
                "或设 CXX=/path/to/g++",
            ],
        ))
    cxx_version = _cxx_version(cxx)

    # XLA FFI include (from jaxlib)
    xla_ffi_include = jax.ffi.include_dir()
    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="未找到 XLA FFI 头 ffi.h（jaxlib 与 CUDA wheel 可能不配套或损坏）。",
            probes=[CandidateProbe("jaxlib:xla/ffi/api/ffi.h", ffi_header, "not-found")],
            remediation=['重装与 CUDA 匹配的 jaxlib：pip install -U "jax[cuda13]"'],
        ))

    # brainevent include
    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="brainevent include 目录缺失（包安装可能损坏）。",
            probes=[CandidateProbe("brainevent/include", be_include, "not-found")],
            remediation=["重装 brainevent：pip install -U --force-reinstall brainevent"],
        ))

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
            stage="host 编译器发现", code="E-CXX",
            summary="未找到 C++ 编译器（CPU 后端需要 g++/clang++）。",
            probes=cxx_probes,
            remediation=[
                "conda 环境：conda install -c conda-forge gxx",
                "Debian/Ubuntu：sudo apt-get install g++",
                "RHEL/Fedora：sudo dnf install gcc-c++",
                "或设 CXX=/path/to/g++",
            ],
        ))
    cxx_version = _cxx_version(cxx)

    xla_ffi_include = jax.ffi.include_dir()
    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="未找到 XLA FFI 头 ffi.h。",
            probes=[CandidateProbe("jaxlib:xla/ffi/api/ffi.h", ffi_header, "not-found")],
            remediation=["重装 jaxlib：pip install -U jax jaxlib"],
        ))

    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise HeaderNotFoundError(render_toolchain_error(
            stage="头文件解析", code="E-HDR",
            summary="brainevent include 目录缺失。",
            probes=[CandidateProbe("brainevent/include", be_include, "not-found")],
            remediation=["重装 brainevent：pip install -U --force-reinstall brainevent"],
        ))

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


def detect_cuda_arch() -> list[str]:
    """Auto-detect GPU compute capabilities via nvidia-smi.

    Returns a list like ``["sm_86"]``.  Falls back to ``["sm_80"]`` if
    detection fails.
    """
    fallback = os.environ.get("BRAINEVENT_COMPUTE_CAPABILITIES", "")
    if fallback:
        return [f"sm_{c.replace('.', '')}" for c in fallback.split(",")]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        result = None

    if result is not None and result.returncode == 0:
        caps = set()
        for line in result.stdout.strip().splitlines():
            cap = line.strip().replace(".", "")
            caps.add(f"sm_{cap}")
        if caps:
            return sorted(caps)

    raise GpuArchDetectionError(render_toolchain_error(
        stage="算力探测", code="E-ARCH",
        summary="无法通过 nvidia-smi 探测 GPU 算力（驱动缺失或 nvidia-smi 不可用）。",
        probes=[CandidateProbe("nvidia-smi", shutil.which("nvidia-smi") or "", "not-found")],
        remediation=[
            "安装/修复 NVIDIA 驱动，使 nvidia-smi 可用",
            "或设 BRAINEVENT_COMPUTE_CAPABILITIES（如 '8.6,8.0'）跳过自动探测",
        ],
    ))
