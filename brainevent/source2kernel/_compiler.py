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
HIPBackend, and NinjaBuild.

Adding a new backend
--------------------
1. Sub-class ``CompilerBackend`` and implement ``compile_source()``.
2. Detect the new toolchain in ``_toolchain.py``.
3. Select the new backend in ``_pipeline.py`` (or call it directly).
"""

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from brainevent._error import CompilationError, KernelToolchainError
from ._toolchain import (
    CppToolchain,
    CudaToolchain,
    cxx_shared_flags,
    cxx_std_flag,
    nvcc_host_pic_flags,
    so_ext,
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CompilerBackend(ABC):
    """Abstract base class for BE compiler backends.

    Each backend encapsulates one compilation toolchain (nvcc, g++, hipcc, …).
    The high-level pipeline in :mod:`brainevent.source2kernel._pipeline`
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
            :func:`~brainevent.source2kernel._codegen.preprocess_source`).
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
# Ninja build helper
# ---------------------------------------------------------------------------

def _find_ninja() -> str | None:
    """Find the ninja binary."""
    return shutil.which("ninja")


class NinjaBuild:
    """Generates and executes a ninja build for CUDA sources.

    Parameters
    ----------
    toolchain : CudaToolchain
        Detected compilation toolchain.
    build_dir : str
        Directory for build artefacts and the ``build.ninja`` file.
    output_name : str
        Base name of the output shared library (without ``.so``).
    gpu_arch : str
        Target GPU architecture.
    extra_cuda_cflags, extra_ldflags, extra_include_paths
        Additional flags and include directories.
    """

    def __init__(
        self,
        toolchain: CudaToolchain,
        build_dir: str,
        output_name: str,
        gpu_arch: str = "sm_80",
        extra_cuda_cflags: list[str] | None = None,
        extra_ldflags: list[str] | None = None,
        extra_include_paths: list[str] | None = None,
        optimization_level: int = 3,
        use_fast_math: bool = False,
    ):
        self.toolchain = toolchain
        self.build_dir = Path(build_dir)
        self.output_name = output_name
        self.gpu_arch = gpu_arch
        self.extra_cuda_cflags = extra_cuda_cflags or []
        self.extra_ldflags = extra_ldflags or []
        self.extra_include_paths = extra_include_paths or []
        self.optimization_level = optimization_level
        self.use_fast_math = use_fast_math
        self._sources: list[str] = []

    def add_source(self, src_path: str) -> None:
        """Register a ``.cu`` source file to compile."""
        self._sources.append(str(src_path))

    @property
    def so_path(self) -> str:
        return str(self.build_dir / f"{self.output_name}{so_ext()}")

    # ------------------------------------------------------------------

    def _cuda_flags(self) -> str:
        flags = [
            f"-arch={self.gpu_arch}",
        ]
        # Platform-aware PIC flags: Linux/macOS need --compiler-options -fPIC;
        # Windows (PE format) does not.
        pic = nvcc_host_pic_flags()
        if pic:
            # In the ninja build file, nvcc passes host options via
            # --compiler-options; quote the sub-flag so the shell doesn't
            # interpret it separately.
            flags += ["--compiler-options", f"'{pic[-1]}'"]
        flags += [
            "--std=c++17",
            f"-O{self.optimization_level}",
            f"-I{self.toolchain.be_include_dir}",
            f"-I{self.toolchain.xla_ffi_include_dir}",
            f"-I{self.toolchain.cuda_include_dir}",
        ]
        if self.use_fast_math:
            flags.append("--use_fast_math")
        for p in self.extra_include_paths:
            flags.append(f"-I{p}")
        flags.extend(self.extra_cuda_cflags)
        return " ".join(flags)

    def _ld_flags(self) -> str:
        return " ".join(self.extra_ldflags)

    def generate(self) -> str:
        """Write ``build.ninja`` and return the output .so path."""
        self.build_dir.mkdir(parents=True, exist_ok=True)
        ninja_path = self.build_dir / "build.ninja"

        objects: list[str] = []
        build_stmts: list[str] = []

        for src in self._sources:
            obj = os.path.splitext(os.path.basename(src))[0] + ".o"
            objects.append(obj)
            build_stmts.append(f"build {obj}: nvcc_compile {src}")

        so_file = f"{self.output_name}{so_ext()}"
        objs = " ".join(objects)
        build_stmts.append(f"build {so_file}: nvcc_link {objs}")

        content = f"""\
# Auto-generated by jax-kernel-bridge — do not edit.
ninja_required_version = 1.3

nvcc = {self.toolchain.nvcc}
cuda_flags = {self._cuda_flags()}
 ld_flags = {self._ld_flags()}

rule nvcc_compile
  command = $nvcc $cuda_flags -c $in -o $out
  description = NVCC $in

rule nvcc_link
  command = $nvcc -shared $in -o $out $ld_flags
  description = LINK $out

{chr(10).join(build_stmts)}

default {so_file}
"""
        ninja_path.write_text(content)
        return self.so_path

    # ------------------------------------------------------------------

    def build(self, verbose: bool = False, workers: int | None = None) -> str:
        """Run ``ninja`` and return the output .so path.

        Raises CompilationError on failure.
        """
        ninja = _find_ninja()
        if ninja is None:
            raise KernelToolchainError(
                "ninja not found. Install with: pip install ninja"
            )

        self.generate()

        cmd = [ninja, "-C", str(self.build_dir)]
        if workers is not None:
            cmd.extend(["-j", str(workers)])
        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise CompilationError(
                "ninja build failed",
                compiler_output=result.stderr + result.stdout,
                command=" ".join(cmd),
            )

        if verbose and result.stdout:
            print(f"ninja output:\n{result.stdout}")

        return self.so_path


# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------

class CUDABackend(CompilerBackend):
    """Compile CUDA sources with nvcc (ninja if available, direct nvcc otherwise).

    Parameters
    ----------
    toolchain : CudaToolchain
        Detected CUDA toolchain (from :func:`~._toolchain.detect_toolchain`).
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
        gpu_arch: str = "sm_80",
        optimization_level: int = 3,
        use_fast_math: bool = False,
        ninja_workers: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Compile preprocessed source to .so (ninja if available, else nvcc)."""
        os.makedirs(build_dir, exist_ok=True)

        # Write source to build dir
        src_path = os.path.join(build_dir, "kernel.cu")
        with open(src_path, "w") as f:
            f.write(source)

        # Try ninja
        try:
            nb = NinjaBuild(
                toolchain=self.toolchain,
                build_dir=build_dir,
                output_name=Path(output_path).stem,
                gpu_arch=gpu_arch,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                optimization_level=optimization_level,
                use_fast_math=use_fast_math,
            )
            nb.add_source(src_path)
            nb.build(verbose=verbose, workers=ninja_workers)
            return output_path
        except Exception as e:
            if verbose:
                print(f"Ninja build not available ({e}), falling back to nvcc")

        # Fallback: direct nvcc
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Build nvcc command — platform-aware flags
        cmd = [
            self.toolchain.nvcc,
            src_path,
            "-shared",
            "-o", output_path,
            f"-arch={gpu_arch}",
            *nvcc_host_pic_flags(),  # -fPIC on Linux/macOS, nothing on Windows
            "--std=c++17",
            f"-O{optimization_level}",
            # Include paths (order matters for override semantics)
            "-I", self.toolchain.be_include_dir,
            "-I", self.toolchain.xla_ffi_include_dir,
            "-I", self.toolchain.cuda_include_dir,
        ]

        if use_fast_math:
            cmd.append("--use_fast_math")

        # Extra include paths
        for p in (extra_include_paths or []):
            cmd.extend(["-I", p])

        # Extra nvcc flags
        cmd.extend(extra_cuda_cflags or [])

        # Extra linker flags
        for flag in (extra_ldflags or []):
            cmd.extend(["--linker-options", flag])

        cmd_str = " ".join(cmd)
        if verbose:
            print(f"nvcc command:\n  {cmd_str}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,  # 5 minutes
        )

        if result.returncode != 0:
            raise CompilationError(
                "nvcc compilation failed",
                compiler_output=result.stderr + result.stdout,
                command=cmd_str,
            )

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
        Detected C++ toolchain (from :func:`~._toolchain.detect_cpp_toolchain`).
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
        with open(src_path, "w") as f:
            f.write(source)

        cmd = [
            toolchain.cxx,
            src_path,
            *cxx_shared_flags(toolchain.cxx),
            "-o", output_path,
            cxx_std_flag(toolchain.cxx),
            "-I", toolchain.be_include_dir,
            "-I", toolchain.xla_ffi_include_dir,
        ]

        for p in (extra_include_paths or []):
            cmd.extend(["-I", p])

        cmd.extend(extra_cflags or [])
        cmd.extend(extra_ldflags or [])

        cmd_str = " ".join(cmd)
        if verbose:
            print(f"C++ command:\n  {cmd_str}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

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

    1. Add ``detect_hip_toolchain()`` to :mod:`~._toolchain` that locates
       ``hipcc`` and the ROCm include directories.
    2. Implement HIP compilation logic in this class using ``hipcc``.
    3. Update :func:`~brainevent.source2kernel._pipeline.load_cuda_inline` (or
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
