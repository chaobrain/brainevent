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

"""Compiler backend abstraction: CompilerBackend ABC, CUDABackend, CPPBackend,
and HIPBackend.

Adding a new backend
--------------------
1. Sub-class ``CompilerBackend`` and implement ``compile_source()``.
2. Detect the new toolchain in ``kernix_toolchain.py``.
3. Select the new backend in ``kernix_pipeline.py`` (or call it directly).
"""

import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from brainevent._error import (
    CompilationError, HostCompilerIncompatibleError,
    UnsupportedArchError,
)
from .kernix_toolchain import (
    CppToolchain, CudaToolchain, cxx_shared_flags, cxx_std_flag,
    gencode_flags, nvcc_host_pic_flags,
)


_HOST_INCOMPAT_SIGNALS = (
    "unsupported gnu version",
    "unsupported clang version",
    "is not supported",
    "no longer supported",
)


def _is_host_incompat(output: str) -> bool:
    low = output.lower()
    return any(sig in low for sig in _HOST_INCOMPAT_SIGNALS)


def _allow_unsupported_compiler() -> bool:
    return os.environ.get("BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _compile_timeout() -> int:
    """Compile subprocess timeout in seconds (``BRAINEVENT_COMPILE_TIMEOUT``)."""
    try:
        return int(os.environ.get("BRAINEVENT_COMPILE_TIMEOUT", "600"))
    except ValueError:
        return 600


def _run(cmd, *, timeout, stage):
    """``subprocess.run`` that maps FileNotFoundError/timeout to CompilationError."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise CompilationError(
            f"compiler executable not found: {cmd[0]}",
            command=" ".join(map(str, cmd)), stage=stage) from e
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") + (e.stderr or "")) if isinstance(e.stdout, str) else ""
        raise CompilationError(
            f"compilation timed out after {timeout}s",
            compiler_output=out, command=" ".join(map(str, cmd)), stage=stage) from e


def _raise_compile_error(output: str, command: str, stage: str) -> None:
    low = output.lower()
    if "unsupported gpu architecture" in low:
        raise UnsupportedArchError(
            "the target GPU architecture is not supported by this nvcc.\n"
            "How to fix:\n"
            "  1) Upgrade the CUDA toolkit: pip install -U 'jax[cuda13]'\n"
            "  2) Or pin a supported arch: brainevent.config.set_compute_capability('8.6')",
            compiler_output=output, command=command, stage=stage)
    if _is_host_incompat(output):
        msg = (
            "host C++ compiler is incompatible with the current CUDA/nvcc.\n"
            "How to fix:\n"
            "  1) Install a supported gcc version and set CXX=/path/to/g++\n"
            "  2) Or set BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1 and retry"
        )
        raise HostCompilerIncompatibleError(msg, compiler_output=output, command=command, stage=stage)
    raise CompilationError("compilation failed", compiler_output=output, command=command, stage=stage)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CompilerBackend(ABC):
    """Abstract base class for BE compiler backends.

    Each backend encapsulates one compilation toolchain (nvcc, g++, hipcc, …).
    The high-level pipeline in :mod:`brainevent._op.kernix_pipeline`
    selects the appropriate backend based on the requested platform.

    Subclasses must implement :meth:`compile_source`.  Optionally they can
    override :meth:`platform_name` to provide a human-readable identifier.
    """

    #: Human-readable platform identifier, e.g. ``"cuda"``, ``"cpu"``, ``"hip"``.
    platform_name: str = "unknown"

    @abstractmethod
    def compile_source(
        self,
        source: str,
        output_path: str,
        build_dir: str,
        *,
        extra_cflags: list[str] | None = None,
        extra_ldflags: list[str] | None = None,
        extra_include_paths: list[str] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> str:
        """Compile preprocessed source code to a shared library.

        Parameters
        ----------
        source : str
            Preprocessed C++/CUDA source (user code + auto-generated FFI
            wrappers, as produced by
            :func:`~brainevent._op.kernix_codegen.preprocess_source`).
        output_path : str
            Desired path for the output shared library.
        build_dir : str
            Directory for intermediate build artefacts.
        extra_cflags : list[str], optional
            Additional compiler flags.
        extra_ldflags : list[str], optional
            Additional linker flags.
        extra_include_paths : list[str], optional
            Additional header search paths.
        verbose : bool
            Print the full compiler command.
        **kwargs
            Backend-specific keyword arguments (e.g. ``gpu_arch`` for CUDA,
            ``optimization_level``, ``use_fast_math``).

        Returns
        -------
        str
            Absolute path to the compiled shared library.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(platform={self.platform_name!r})"


# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------

class CUDABackend(CompilerBackend):
    """Compile CUDA sources directly with nvcc.

    Parameters
    ----------
    toolchain : CudaToolchain
        Detected CUDA toolchain (from :func:`~.kernix_toolchain.detect_toolchain`).
    """

    platform_name = "cuda"

    def __init__(self, toolchain: CudaToolchain) -> None:
        self.toolchain = toolchain

    def compile_source(
        self,
        source: str,
        output_path: str,
        build_dir: str,
        *,
        extra_cuda_cflags: list[str] | None = None,
        extra_ldflags: list[str] | None = None,
        extra_include_paths: list[str] | None = None,
        verbose: bool = False,
        gpu_arch: "str | list[str]" = "sm_80",
        optimization_level: int = 3,
        use_fast_math: bool = False,
        **kwargs: Any,
    ) -> str:
        """Compile preprocessed source to .so directly with nvcc."""
        os.makedirs(build_dir, exist_ok=True)
        arches = [gpu_arch] if isinstance(gpu_arch, str) else list(gpu_arch)

        # Write source to build dir (utf-8: nvcc expects utf-8)
        src_path = os.path.join(build_dir, "kernel.cu")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source)

        return self._compile_direct(
            src_path, output_path, arches, extra_cuda_cflags, extra_ldflags,
            extra_include_paths, verbose, optimization_level, use_fast_math)

    def _compile_direct(
        self, src_path, output_path, arches, extra_cuda_cflags, extra_ldflags,
        extra_include_paths, verbose, optimization_level, use_fast_math,
    ) -> str:
        """Compile + link a single .cu directly with nvcc."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cmd = [
            self.toolchain.nvcc,
            src_path,
            "-shared",
            "-o", output_path,
            *gencode_flags(arches),
            *nvcc_host_pic_flags(),  # -fPIC on Linux/macOS, nothing on Windows
            "--std=c++17",
            f"-O{optimization_level}",
            "-ccbin", self.toolchain.cxx,
        ]
        if _allow_unsupported_compiler():
            cmd.append("-allow-unsupported-compiler")
        # Include paths (order matters for override semantics)
        cmd += ["-I", self.toolchain.brainevent_include_dir,
                "-I", self.toolchain.xla_ffi_include_dir]
        for inc in self.toolchain.cuda_include_dirs:
            cmd += ["-I", inc]

        if use_fast_math:
            cmd.append("--use_fast_math")

        for p in (extra_include_paths or []):
            cmd.extend(["-I", p])

        cmd.extend(extra_cuda_cflags or [])

        for flag in (extra_ldflags or []):
            cmd.extend(["--linker-options", flag])

        cmd_str = " ".join(cmd)
        if verbose:
            print(f"nvcc command:\n  {cmd_str}")

        result = _run(cmd, timeout=_compile_timeout(), stage="compile")

        if result.returncode != 0:
            _raise_compile_error(result.stderr + result.stdout, cmd_str, stage="compile")

        if verbose and result.stderr:
            print(f"nvcc warnings:\n{result.stderr}")

        return output_path


# ---------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------

class CPPBackend(CompilerBackend):
    """Compile C++ sources with g++ / clang++.

    Parameters
    ----------
    toolchain : CppToolchain
        Detected C++ toolchain (from :func:`~.kernix_toolchain.detect_cpp_toolchain`).
    """

    platform_name = "cpu"

    def __init__(self, toolchain: CppToolchain) -> None:
        self.toolchain = toolchain

    def compile_source(
        self,
        source: str,
        output_path: str,
        build_dir: str,
        *,
        extra_cflags: list[str] | None = None,
        extra_ldflags: list[str] | None = None,
        extra_include_paths: list[str] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> str:
        toolchain = self.toolchain

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        build_dir = os.path.dirname(os.path.abspath(output_path))
        src_path = os.path.join(build_dir, "kernel.cpp")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source)

        cmd = [
            toolchain.cxx,
            src_path,
            *cxx_shared_flags(toolchain.cxx),
            "-o", output_path,
            cxx_std_flag(toolchain.cxx),
            "-I", toolchain.brainevent_include_dir,
            "-I", toolchain.xla_ffi_include_dir,
        ]

        for p in (extra_include_paths or []):
            cmd.extend(["-I", p])

        cmd.extend(extra_cflags or [])
        cmd.extend(extra_ldflags or [])

        cmd_str = " ".join(cmd)
        if verbose:
            print(f"C++ command:\n  {cmd_str}")

        result = _run(cmd, timeout=_compile_timeout(), stage="compile")

        if result.returncode != 0:
            raise CompilationError(
                "C++ compilation failed",
                compiler_output=result.stderr + result.stdout,
                command=cmd_str,
            )

        if verbose and result.stderr:
            print(f"Compiler warnings:\n{result.stderr}")

        return output_path


# ---------------------------------------------------------------------------
# HIP backend (stub — not yet implemented)
# ---------------------------------------------------------------------------

class HIPBackend(CompilerBackend):
    """Compile HIP sources for AMD GPUs (stub — not yet implemented).

    To implement HIP support:

    1. Add ``detect_hip_toolchain()`` to :mod:`~.kernix_toolchain` that locates
       ``hipcc`` and the ROCm include directories.
    2. Implement HIP compilation logic in this class using ``hipcc``.
    3. Update :func:`~brainevent._op.kernix_pipeline.load_cuda_inline` (or
       add ``load_hip_inline``) to select :class:`HIPBackend` when
       ``platform="hip"`` is requested.
    """

    platform_name = "hip"

    def compile_source(self, source: str, output_path: str, build_dir: str,
                       **kwargs: Any) -> str:
        raise NotImplementedError(
            "HIP backend is not yet implemented.  "
            "See the class docstring for implementation guidance."
        )
