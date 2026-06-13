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

"""High-level compilation pipeline and diagnostics.

Public API: load_cuda_inline, load_cuda_file, load_cuda_dir,
            load_cpp_inline, load_cpp_file,
            clear_cache, set_cache_dir, get_cache_dir,
            print_diagnostics.
"""

import glob
import os
import shutil
import sys
import uuid
from pathlib import Path

import jax
import jaxlib

from brainevent._error import KernelError
from brainevent._version import __version__
from .kernix_cache import CompilationCache
from .kernix_codegen import (
    FunctionSpec, infer_arg_spec_from_source, normalize_tokens,
    parse_arg_spec, parse_annotations, preprocess_source, resolve_bare_attr_types,
)
from .kernix_compiler import CPPBackend, CUDABackend
from .kernix_runtime import CompiledModule, _REGISTERED_TARGETS, register_ffi_target
from .kernix_toolchain import (
    collect_toolchain_diagnostics, detect_cpp_toolchain, detect_cuda_arch,
    detect_cuda_toolchain, get_compute_capabilities, resolve_compute_capabilities,
    so_ext,
)

# Shared cache instance
_cache = CompilationCache()

# In-process map of FFI target name -> the so_path it was registered from.
# Re-loading the same artefact (cache hit, same target_prefix) is a no-op, but
# loading the same source under a *different* target_prefix still registers the
# new target names.  A different artefact claiming an existing target name
# surfaces a KernelRegistrationError (via register_ffi_target).
_REGISTERED_TARGET_SO: dict[str, str] = {}


def _join_sources(sources: str | list[str]) -> str:
    """Concatenate a list of sources, or return a single source unchanged."""
    return "\n\n".join(sources) if isinstance(sources, list) else sources


def _cache_header_paths(toolchain) -> list[str]:
    """Resolve the injected headers whose byte-contents affect the build.

    Every ``brainevent`` header (``ffi_compat.h``, ``tensor.h``, ``check.h``,
    ``cuda_common.h``, ...) can be pulled into a generated translation unit, so
    a content change in *any* of them must rebuild — hashing only ``ffi_compat.h``
    would miss, e.g., a ``check.h`` semantics change (abort -> throw) when the
    ``brainevent`` version is unchanged (editable installs).  jaxlib's
    ``xla/ffi/api/ffi.h`` is hashed too so an ABI bump rebuilds.

    Parameters
    ----------
    toolchain : CudaToolchain or CppToolchain
        A detected toolchain exposing ``brainevent_include_dir`` and
        ``xla_ffi_include_dir``.

    Returns
    -------
    list of str
        Existing header paths to byte-hash (missing files are skipped; the
        cache key tolerates absent entries).  The order is irrelevant — the
        cache key sorts the paths.
    """
    be_inc = toolchain.brainevent_include_dir
    candidates: list[str] = []
    # All brainevent headers: both the ``brainevent/`` subdir and any top-level
    # headers (cuda_common.h / curand_common.h live one level up).
    candidates.extend(glob.glob(os.path.join(be_inc, "brainevent", "*.h")))
    candidates.extend(glob.glob(os.path.join(be_inc, "*.h")))
    # jaxlib FFI ABI header.
    candidates.append(os.path.join(toolchain.xla_ffi_include_dir, "xla", "ffi", "api", "ffi.h"))
    return [p for p in candidates if os.path.isfile(p)]


def _build_specs(
    functions: dict[str, list[str]], user_source: str
) -> list[FunctionSpec]:
    """Parse each function's arg_spec: normalize aliases → resolve bare attrs → parse."""
    specs: list[FunctionSpec] = []
    for func_name, tokens in functions.items():
        tokens = normalize_tokens(tokens)
        tokens = resolve_bare_attr_types(tokens, func_name, user_source)
        specs.append(parse_arg_spec(func_name, tokens))
    return specs


def _register_targets(
    specs: list[FunctionSpec],
    module: CompiledModule,
    so_path: str,
    name: str,
    target_prefix: str | None,
    platform: str,
) -> None:
    """Register each spec as a JAX FFI target, deduped per target name.

    A target already registered from the same *so_path* is skipped (idempotent
    re-load).  The same source under a different *target_prefix* yields new
    target names, which are registered.  A name already taken by a *different*
    artefact raises ``KernelRegistrationError`` from ``register_ffi_target``.
    """
    prefix = target_prefix or name
    for spec in specs:
        target = f"{prefix}.{spec.name}"
        if _REGISTERED_TARGET_SO.get(target) == so_path:
            continue
        register_ffi_target(target, module, spec.name, platform=platform)
        _REGISTERED_TARGET_SO[target] = so_path


def set_cache_dir(path: str | Path) -> None:
    """Set the compilation cache directory.

    All subsequent compilations will use this directory for caching.
    Existing cached artefacts in the old directory are **not** moved.

    Parameters
    ----------
    path : str or Path
        New cache directory path.  Created if it does not exist.
    """
    global _cache
    _cache = CompilationCache(base_dir=str(path))


def get_cache_dir() -> str:
    """Return the current compilation cache directory path."""
    return str(_cache.base_dir)


def load_cuda_inline(
    name: str,
    cuda_sources: str | list[str],
    functions: dict[str, list[str]] | None = None,
    *,
    extra_cuda_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    extra_include_paths: list[str] | None = None,
    build_directory: str | None = None,
    verbose: bool = False,
    compute_capability: str | None = None,
    force_rebuild: bool = False,
    auto_register: bool = True,
    target_prefix: str | None = None,
    optimization_level: int = 3,
    use_fast_math: bool = False,
    allow_cuda_graph: bool = True,
) -> CompiledModule:
    """Compile inline CUDA source and load the resulting module.

    Parameters
    ----------
    name : str
        Module name (used for caching and FFI target naming).
    cuda_sources : str or list[str]
        CUDA C++ source code.  Multiple strings are concatenated.
    functions : dict[str, list[str]] or None
        Mapping from function name to its arg_spec token list.
        Example: ``{"vector_add": ["arg", "arg", "ret", "stream"]}``

        If ``None``, functions are discovered from ``// @BE function_name``
        annotations in the source code.  The arg_spec is auto-inferred
        from the C++ signature.
    extra_cuda_cflags, extra_ldflags, extra_include_paths
        Additional compilation/linking flags.
    build_directory : str, optional
        Override the build directory.
    verbose : bool
        Print detailed compilation output.
    compute_capability : str, optional
        GPU architecture (e.g. ``"sm_86"``).  Auto-detected if ``None``.
    force_rebuild : bool
        Skip cache and recompile.
    auto_register : bool
        Automatically register each function as a JAX FFI target with
        name ``"<target_prefix>.<func_name>"`` (or ``"<name>.<func>"``
        if *target_prefix* is ``None``).
    target_prefix : str, optional
        Prefix for auto-registered FFI target names.
    optimization_level : int
        Compiler optimization level passed as ``-O<n>`` to nvcc (0–3).
        Applies to both host code and device PTX generation.  Default: ``3``.
    use_fast_math : bool
        Pass ``--use_fast_math`` to nvcc.  Enables approximate division/sqrt,
        flush-to-zero for denormals, and fused multiply-add.  Can give
        10–30 % speed-up on FP-heavy kernels at the cost of reduced IEEE
        precision.  Default: ``False``.
    allow_cuda_graph : bool
        Register kernels with the ``COMMAND_BUFFER_COMPATIBLE`` XLA trait so
        they can be captured and replayed by JAX's CUDA-graph optimisation.
        Eliminates per-call CPU launch overhead inside ``jax.lax.fori_loop``
        or repeated ``jax.jit`` calls.  Set to ``False`` only for kernels
        with host-side side effects during replay.  Default: ``True``.

    Returns
    -------
    CompiledModule
    """
    user_source = _join_sources(cuda_sources)

    # Discover functions from annotations if not provided
    if functions is None:
        functions = parse_annotations(user_source)
    if not functions:
        raise KernelError(
            "No functions found. Pass functions={...} or add '// @BE <name>' "
            "annotations to the source."
        )

    # Detect toolchain
    toolchain = detect_cuda_toolchain()

    # Resolve target architecture(s).  Pure auto-detection compiles for the
    # default device only (with PTX forward-compat); an explicit pin (arg /
    # config / env) honours the full list (fat binary).
    explicit_pin = (
        compute_capability is not None
        or get_compute_capabilities() is not None
        or bool(os.environ.get("BRAINEVENT_COMPUTE_CAPABILITIES", "").strip())
    )
    resolved = resolve_compute_capabilities(compute_capability)
    gpu_arches = resolved if explicit_pin else resolved[:1]
    arch_key = "+".join(gpu_arches)

    # Compute cache key (includes optimization settings so changing them
    # rebuilds; also the include search path, injected-header bytes, and the
    # jaxlib FFI ABI version — see CompilationCache.cache_key).
    cache_key = _cache.cache_key(
        source=user_source,
        arch=arch_key,
        cxx_version=f"{toolchain.nvcc_version}|{toolchain.cxx_version}",
        extra_cflags=(
            (extra_cuda_cflags or [])
            + [f"-O{optimization_level}"]
            + (["--use_fast_math"] if use_fast_math else [])
            + [f"--be-allow-cuda-graph={int(allow_cuda_graph)}"]
        ),
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        header_paths=_cache_header_paths(toolchain),
    )

    # Check cache
    cached_so = None if force_rebuild else _cache.lookup(name, cache_key)

    specs = _build_specs(functions, user_source)

    if cached_so is not None:
        so_path = str(cached_so)
    else:
        # Preprocess: inject headers + generate FFI wrappers
        full_source = preprocess_source(user_source, specs, allow_cuda_graph=allow_cuda_graph)

        # Build in a unique temp dir, then atomically publish into the cache so
        # concurrent builds never expose a partial .so.  A user-supplied
        # build_directory is honoured as-is (and not cleaned up).
        if build_directory:
            build_dir, cleanup = build_directory, False
        else:
            build_dir = os.path.join(
                str(_cache.base_dir), f".build-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            cleanup = True
        os.makedirs(build_dir, exist_ok=True)
        so_path = os.path.join(build_dir, f"{name}{so_ext()}")

        try:
            # Compile via the CUDA backend (direct nvcc).
            CUDABackend(toolchain).compile_source(
                full_source,
                so_path,
                build_dir,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                verbose=verbose,
                gpu_arch=gpu_arches,
                optimization_level=optimization_level,
                use_fast_math=use_fast_math,
            )
            # A user-supplied build_directory is the caller's: copy (don't move)
            # the artefact into the cache so their file stays put.
            so_path = str(_cache.store(
                name, cache_key, so_path, source_is_user_dir=bool(build_directory)))
        finally:
            if cleanup:
                shutil.rmtree(build_dir, ignore_errors=True)

    # Load module
    module = CompiledModule(so_path, list(functions.keys()))

    if auto_register:
        _register_targets(specs, module, so_path, name, target_prefix, platform="CUDA")

    return module


def load_cuda_file(
    filepath: str | Path,
    functions: dict[str, list[str]] | None = None,
    *,
    name: str | None = None,
    **kwargs,
) -> CompiledModule:
    """Compile a single ``.cu`` file and load the resulting module.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.cu`` file.
    functions : dict[str, list[str]] or None
        Function name → arg_spec mapping (same as ``load_cuda_inline``).
        If ``None``, discovered from ``// @BE`` annotations.
    name : str, optional
        Module name.  Defaults to the file stem.
    **kwargs
        Forwarded to ``load_cuda_inline``.
    """
    filepath = Path(filepath)
    if name is None:
        name = filepath.stem
    source = filepath.read_text(encoding="utf-8")
    return load_cuda_inline(name=name, cuda_sources=source, functions=functions, **kwargs)


def load_cuda_dir(
    directory: str | Path,
    functions: dict[str, list[str]] | None = None,
    *,
    name: str | None = None,
    file_patterns: list[str] | None = None,
    **kwargs,
) -> CompiledModule:
    """Compile all CUDA files in a directory and load the resulting module.

    Parameters
    ----------
    directory : str or Path
        Directory containing ``.cu`` / ``.cuh`` files.
    functions : dict[str, list[str]] or None
        Function name → arg_spec mapping.  If ``None``, discovered from
        ``// @BE`` annotations.
    name : str, optional
        Module name.  Defaults to the directory name.
    file_patterns : list[str], optional
        Glob patterns for source files (default: ``["*.cu"]``).
    **kwargs
        Forwarded to ``load_cuda_inline``.
    """
    directory = Path(directory)
    if name is None:
        name = directory.name
    patterns = file_patterns or ["*.cu"]

    sources: list[str] = []
    for pat in patterns:
        for path in sorted(directory.glob(pat)):
            sources.append(path.read_text(encoding="utf-8"))

    if not sources:
        raise KernelError(f"No source files matching {patterns} found in {directory}")

    return load_cuda_inline(name=name, cuda_sources=sources, functions=functions, **kwargs)


def load_cpp_inline(
    name: str,
    cpp_sources: str | list[str],
    functions: dict[str, list[str]] | list[str] | None = None,
    *,
    extra_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    extra_include_paths: list[str] | None = None,
    build_directory: str | None = None,
    verbose: bool = False,
    force_rebuild: bool = False,
    auto_register: bool = True,
    target_prefix: str | None = None,
) -> CompiledModule:
    """Compile inline C++ source for **CPU** (or CUDA) and load the module.

    This is the CPU counterpart of :func:`load_cuda_inline`.  It uses the system
    C++ compiler (``g++`` / ``clang++``) instead of ``nvcc``, so **CUDA is
    not required**.

    Parameters
    ----------
    name : str
        Module name (used for caching and FFI target naming).
    cpp_sources : str or list[str]
        C++ source code.  Multiple strings are concatenated.
    functions : dict, list, or None
        *Dict form* (explicit): ``{"func": ["arg", "ret", ...]}`` — same
        arg_spec tokens as :func:`load_cuda_inline`.

        *List form* (auto-detect): ``["func"]`` — the arg_spec is inferred
        from the C++ signature.  ``const BE::Tensor`` → ``"arg"``,
        non-const ``BE::Tensor`` → ``"ret"``.

        *None* (annotation): functions are discovered from
        ``// @BE function_name`` annotations in the source code.
    extra_cflags, extra_ldflags, extra_include_paths
        Additional compiler / linker flags and include paths.
    build_directory : str, optional
        Override the build directory.
    verbose : bool
        Print the full compiler command.
    force_rebuild : bool
        Skip cache and recompile.
    auto_register : bool
        Automatically register FFI targets as ``"<prefix>.<func>"``.
    target_prefix : str, optional
        Prefix for auto-registered FFI targets.  Defaults to *name*.

    Returns
    -------
    CompiledModule
    """

    user_source = _join_sources(cpp_sources)

    # Resolve functions → dict[str, list[str]]
    if functions is None:
        functions = parse_annotations(user_source)
    elif isinstance(functions, list):
        functions = {fn: infer_arg_spec_from_source(user_source, fn) for fn in functions}
    if not functions:
        raise KernelError(
            "No functions found. Pass functions={...} or add '// @BE <name>' "
            "annotations to the source."
        )

    specs = _build_specs(functions, user_source)

    # Detect toolchain (CPU — no CUDA needed)
    toolchain = detect_cpp_toolchain()

    # Cache key (include search path, injected-header bytes, and the jaxlib
    # FFI ABI version participate — see CompilationCache.cache_key).
    cache_key = _cache.cache_key(
        source=user_source,
        arch="cpu",
        cxx_version=toolchain.cxx_version,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        header_paths=_cache_header_paths(toolchain),
    )

    cached_so = None if force_rebuild else _cache.lookup(name, cache_key)

    if cached_so is not None:
        so_path = str(cached_so)
    else:
        full_source = preprocess_source(user_source, specs, platform="cpu")

        if build_directory:
            build_dir, cleanup = build_directory, False
        else:
            build_dir = os.path.join(
                str(_cache.base_dir), f".build-{os.getpid()}-{uuid.uuid4().hex[:8]}")
            cleanup = True
        os.makedirs(build_dir, exist_ok=True)
        so_path = os.path.join(build_dir, f"{name}{so_ext()}")

        try:
            CPPBackend(toolchain).compile_source(
                full_source,
                so_path,
                build_dir,
                extra_cflags=extra_cflags,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                verbose=verbose,
            )
            # A user-supplied build_directory is the caller's: copy (don't move)
            # the artefact into the cache so their file stays put.
            so_path = str(_cache.store(
                name, cache_key, so_path, source_is_user_dir=bool(build_directory)))
        finally:
            if cleanup:
                shutil.rmtree(build_dir, ignore_errors=True)

    module = CompiledModule(so_path, list(functions.keys()))

    if auto_register:
        _register_targets(specs, module, so_path, name, target_prefix, platform="cpu")

    return module


def load_cpp_file(
    filepath: str | Path,
    functions: dict[str, list[str]] | list[str] | None = None,
    *,
    name: str | None = None,
    **kwargs,
) -> CompiledModule:
    """Compile a single ``.cpp`` / ``.cc`` file for CPU and load the module.

    Parameters
    ----------
    filepath : str or Path
        Path to the C++ source file.
    functions : dict, list, or None
        Same as :func:`load_cpp_inline`.
    name : str, optional
        Module name.  Defaults to the file stem.
    **kwargs
        Forwarded to :func:`load_cpp_inline`.
    """
    filepath = Path(filepath)
    if name is None:
        name = filepath.stem
    source = filepath.read_text(encoding="utf-8")
    return load_cpp_inline(name=name, cpp_sources=source, functions=functions, **kwargs)


def clear_cache(name: str | None = None) -> int:
    """Clear the compilation cache.  Returns number of entries removed."""
    return _cache.clear(name)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics() -> None:
    """Print a summary of the brainevent compilation environment."""

    print(f"brainevent v{__version__}")
    print(f"Python: {sys.version}")

    # JAX / jaxlib
    print(f"JAX: {jax.__version__}")
    print(f"jaxlib: {jaxlib.__version__}")
    print(f"XLA FFI include: {jax.ffi.include_dir()}")

    # Toolchain (single-source snapshot)
    for k, v in collect_toolchain_diagnostics().items():
        print(f"{k}: {v}")
    try:
        archs = detect_cuda_arch()
        print(f"GPU architectures: {', '.join(archs)}")
    except Exception as e:
        print(f"GPU architectures: ERROR ({e.__class__.__name__})")

    # Cache
    entries, total_bytes = _cache.size()
    if total_bytes > 1_000_000:
        size_str = f"{total_bytes / 1_000_000:.1f} MB"
    elif total_bytes > 1_000:
        size_str = f"{total_bytes / 1_000:.1f} KB"
    else:
        size_str = f"{total_bytes} bytes"
    print(f"Cache: {_cache.base_dir} ({entries} entries, {size_str})")

    # Registered targets
    print(f"Registered FFI targets: {len(_REGISTERED_TARGETS)}")
    for t in sorted(_REGISTERED_TARGETS):
        print(f"  - {t}")
