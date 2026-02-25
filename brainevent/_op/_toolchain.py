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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax

from brainevent._error import KernelToolchainError


@dataclass(frozen=True)
class CudaToolchain:
    """Immutable description of available CUDA compilation tools."""
    nvcc: str
    cxx: str
    cuda_home: str
    cuda_include_dir: str
    xla_ffi_include_dir: str
    be_include_dir: str
    nvcc_version: str = ""


@dataclass(frozen=True)
class CppToolchain:
    """Immutable description of the CPU/C++ compilation tools (no CUDA required)."""
    cxx: str
    xla_ffi_include_dir: str
    be_include_dir: str
    cxx_version: str = ""


def detect_toolchain() -> CudaToolchain:
    """Auto-detect nvcc, C++ compiler, and include paths.

    Raises ToolchainError if essential tools are missing.
    """
    # --- nvcc ---
    nvcc = os.environ.get("JKB_NVCC_PATH") or shutil.which("nvcc")
    if nvcc is None:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(candidate):
            nvcc = candidate
    if nvcc is None:
        raise KernelToolchainError(
            "Cannot find nvcc. Ensure CUDA Toolkit is installed and nvcc is "
            "on PATH, or set JKB_NVCC_PATH / CUDA_HOME."
        )

    # --- CUDA home & include ---
    cuda_home = os.environ.get("CUDA_HOME", "")
    if not cuda_home:
        # Derive from nvcc path: .../bin/nvcc → ...
        cuda_home = str(Path(nvcc).resolve().parent.parent)
    cuda_include = os.path.join(cuda_home, "include")

    # --- nvcc version ---
    nvcc_version = ""
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
        for line in proc.stdout.splitlines():
            if "release" in line.lower():
                nvcc_version = line.strip()
                break
    except Exception:
        pass

    # --- C++ compiler ---
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        raise KernelToolchainError("Cannot find a C++ compiler. Install g++ or set CXX.")

    # --- XLA FFI include dir (from jaxlib) ---
    xla_ffi_include = jax.ffi.include_dir()

    # Verify the critical header exists
    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise KernelToolchainError(
            f"XLA FFI header not found at {ffi_header}. "
            f"jaxlib include dir: {xla_ffi_include}"
        )

    # --- BE include dir (brainevent/include/, shipped with this package) ---
    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise KernelToolchainError(
            f"BE include directory not found at {be_include}. "
            "Package may be installed incorrectly."
        )

    return CudaToolchain(
        nvcc=nvcc,
        cxx=cxx,
        cuda_home=cuda_home,
        cuda_include_dir=cuda_include,
        xla_ffi_include_dir=xla_ffi_include,
        be_include_dir=be_include,
        nvcc_version=nvcc_version,
    )


def detect_cpp_toolchain() -> CppToolchain:
    """Auto-detect a C++ compiler and include paths for CPU-only compilation.

    Does not require CUDA to be installed.

    Raises ToolchainError if essential tools are missing.
    """
    # --- C++ compiler ---
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        raise KernelToolchainError("Cannot find a C++ compiler. Install g++ or clang++, or set CXX.")

    # --- cxx version string ---
    cxx_version = ""
    try:
        proc = subprocess.run(
            [cxx, "--version"], capture_output=True, text=True, timeout=10
        )
        cxx_version = proc.stdout.splitlines()[0].strip() if proc.stdout else ""
    except Exception:
        pass

    # --- XLA FFI include dir (from jaxlib) ---
    try:
        import jax.ffi
        xla_ffi_include = jax.ffi.include_dir()
    except Exception as e:
        raise KernelToolchainError(
            f"Cannot determine XLA FFI include directory: {e}. "
            "Ensure jax and jaxlib are installed."
        ) from e

    ffi_header = os.path.join(xla_ffi_include, "xla", "ffi", "api", "ffi.h")
    if not os.path.isfile(ffi_header):
        raise KernelToolchainError(
            f"XLA FFI header not found at {ffi_header}. "
            f"jaxlib include dir: {xla_ffi_include}"
        )

    # --- BE include dir ---
    be_include = str(Path(__file__).resolve().parent.parent / "include")
    if not os.path.isdir(be_include):
        raise KernelToolchainError(f"BE include directory not found at {be_include}.")

    return CppToolchain(
        cxx=cxx,
        xla_ffi_include_dir=xla_ffi_include,
        be_include_dir=be_include,
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
    fallback = os.environ.get("JKB_COMPUTE_CAPABILITIES", "")
    if fallback:
        return [f"sm_{c.replace('.', '')}" for c in fallback.split(",")]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            caps = set()
            for line in result.stdout.strip().splitlines():
                cap = line.strip().replace(".", "")
                caps.add(f"sm_{cap}")
            if caps:
                return sorted(caps)
    except FileNotFoundError:
        pass

    return ["sm_80"]
