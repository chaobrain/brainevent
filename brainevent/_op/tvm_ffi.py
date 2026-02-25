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

# -*- coding: utf-8 -*-

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Sequence

from brainevent._error import TVMFFINotInstalledError, TVMModuleAlreadyRegisteredError

tvmffi_installed = importlib.util.find_spec('jax_tvm_ffi') is not None

# Try to import TVM FFI - will fail gracefully if not available
if tvmffi_installed:
    try:
        import jax_tvm_ffi
        import tvm_ffi.cpp
    except:
        tvmffi_installed = False

__all__ = [
    'register_tvm_cuda_kernels',
    'register_tvm_cuda_from_file',
]

# Global cache: tracks compiled CUDA modules and the registration signature
# (source hash + function names) associated with each module name.
_registered_tvm_modules: dict = dict()
_registered_tvm_module_signatures: dict = dict()


def _make_tvm_module_signature(source_code: str, functions: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Create a stable signature for TVM module registration requests."""
    source_hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
    # Function order does not change the logical FFI target set.
    function_names = tuple(sorted(functions))
    return source_hash, function_names


def register_tvm_cuda_kernels(
    source_code: str,
    module: str,
    functions: Sequence[str],
    arg_spec: Sequence[str] = ("args", "rets", "ctx.stream"),
    include_dir: str | Path | None = None,
):
    """Compile CUDA source code and register the resulting kernels with JAX FFI.

    Uses the TVM FFI infrastructure (``jax_tvm_ffi`` and ``tvm_ffi.cpp``)
    to compile inline CUDA source and register each resulting function as
    a JAX FFI target on the GPU platform.

    A per-process cache tracks registered module signatures.  If *module*
    has already been registered with the same source/function signature,
    this function returns the cached module and skips recompilation.
    If the same *module* name is requested with a different signature,
    :class:`~brainevent._error.TVMModuleAlreadyRegisteredError` is raised.

    Parameters
    ----------
    source_code : str
        CUDA C/C++ source code containing the kernel implementations.
    module : str
        Module name under which the compiled kernels are registered.
        Each kernel is registered as ``"<module>.<function_name>"``.
        Must be unique within the process.
    functions : Sequence of str
        Names of the functions (entry points) to extract from the
        compiled module and register with JAX FFI.
    include_dir : str or Path or None, optional
        Directory path to add to the include search path during compilation.
        This allows ``#include`` directives in the source to find headers.

    Returns
    -------
    object
        The compiled CUDA module object.  If *module* has already been
        registered with an identical signature in this process, the
        cached module is returned.

    Raises
    ------
    TVMFFINotInstalledError
        If ``jax_tvm_ffi`` or ``tvm_ffi.cpp`` is not installed.
    TVMModuleAlreadyRegisteredError
        If *module* was already registered in this process with different
        CUDA source code or a different set of function names.
    ValueError
        If *source_code* is not a string, *module* is not a string, or
        *functions* is not a sequence of strings.

    See Also
    --------
    XLACustomKernel.def_tvmffi_kernel : Register a TVM FFI kernel with
        an ``XLACustomKernel``.

    Notes
    -----
    The compiled kernels are registered as JAX FFI targets using the
    naming convention ``"<module>.<function_name>"`` and are available
    on the ``"gpu"`` platform.

    Examples
    --------
    .. code-block:: python

        >>> cuda_src = '''
        ... extern "C" void add_arrays(float* a, float* b, float* out, int n) {
        ...     int i = threadIdx.x + blockIdx.x * blockDim.x;
        ...     if (i < n) out[i] = a[i] + b[i];
        ... }
        ... '''
        >>> register_tvm_cuda_kernels(cuda_src, 'my_kernels', ['add_arrays'])
    """

    if not tvmffi_installed:
        raise TVMFFINotInstalledError(
            "jax_tvm_ffi is not installed. "
            "Install it with: pip install jax-tvm-ffi"
        )

    if not isinstance(source_code, str):
        raise ValueError("source_code must be a string")
    if not isinstance(module, str):
        raise ValueError("module must be a string")
    if not isinstance(functions, Sequence) or not all(isinstance(f, str) for f in functions):
        raise ValueError("functions must be a sequence of strings")

    requested_signature = _make_tvm_module_signature(source_code, functions)

    # Return cached module only when the registration signature matches.
    if module in _registered_tvm_modules:
        cached_signature = _registered_tvm_module_signatures[module]
        if cached_signature != requested_signature:
            raise TVMModuleAlreadyRegisteredError(
                f"TVM CUDA module '{module}' is already registered with a different "
                f"source/functions signature. Use a unique module name per kernel definition."
            )
        return _registered_tvm_modules[module]

    # Prepare compilation options
    compile_options = {}
    if include_dir:
        include_dir_path = Path(include_dir)
        if not include_dir_path.is_dir():
            raise ValueError(f"Include directory not found: {include_dir}")
        compile_options['extra_include_paths'] = [str(include_dir_path)]

    # Compile CUDA module
    _cuda_module = tvm_ffi.cpp.load_inline(
        name=module,
        cuda_sources=source_code,
        functions=functions,
        **compile_options
    )

    # Register each kernel with JAX FFI
    for name in functions:
        jax_tvm_ffi.register_ffi_target(
            name=f"{module}.{name}",
            function=getattr(_cuda_module, name),
            arg_spec=list(arg_spec),
            platform="gpu",
            allow_cuda_graph=True,
            pass_owned_tensor=False,
            use_last_output_for_alloc_workspace=False,
        )

    # Mark this module as registered so future calls can safely reuse it.
    _registered_tvm_modules[module] = _cuda_module
    _registered_tvm_module_signatures[module] = requested_signature
    return _cuda_module


def _parse_tvm_entry_functions(source_code: str) -> list:
    """Parse TVM FFI entry function names from CUDA source code.

    Discovers entry points via two mechanisms (results are merged,
    duplicates removed, source order preserved):

    1. **Explicit functions** -- top-level ``void`` functions whose
       parameter list contains ``tvm::ffi::TensorView``.
       ``__device__`` and ``__global__`` functions are excluded
       automatically because they begin with a kernel qualifier rather
       than plain ``void``.

    2. **Annotation comments** -- lines of the form::

           // @tvm_ffi function_name

       This allows macro-generated FFI entry points to be registered
       without writing the function signature explicitly in the source.
       The macro expands the ``void function_name(...)`` body, and the
       annotation tells the parser about it.

    Parameters
    ----------
    source_code : str
        CUDA C/C++ source code to scan.

    Returns
    -------
    list of str
        Discovered entry-point function names, in source order.

    Raises
    ------
    TypeError
        If *source_code* is not a string.
    ValueError
        If a ``void funcname(`` pattern is found but the opening
        parenthesis is unmatched (malformed CUDA source).
    """
    if not isinstance(source_code, str):
        raise TypeError(
            f"source_code must be a str, got {type(source_code).__name__}"
        )

    functions = []
    seen = set()

    # --- Method 1: explicit void functions at column 0 ---
    func_pattern = re.compile(r'^void\s+(\w+)\s*\(', re.MULTILINE)
    for match in func_pattern.finditer(source_code):
        func_name = match.group(1)
        # Walk forward from '(' to find the matching ')' and inspect params.
        paren_start = match.end() - 1  # index of the opening '('
        depth = 0
        pos = paren_start
        while pos < len(source_code):
            if source_code[pos] == '(':
                depth += 1
            elif source_code[pos] == ')':
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        if depth != 0:
            raise ValueError(
                f"Unmatched parenthesis in CUDA source while parsing "
                f"function '{func_name}' near offset {paren_start}. "
                "The source file may be truncated or malformed."
            )
        params = source_code[paren_start:pos + 1]
        if 'tvm::ffi::TensorView' in params:
            if func_name not in seen:
                functions.append(func_name)
                seen.add(func_name)

    # --- Method 2: annotation comments  // @tvm_ffi name ---
    annot_pattern = re.compile(r'^\s*//\s*@tvm_ffi\s+(\w+)', re.MULTILINE)
    for match in annot_pattern.finditer(source_code):
        func_name = match.group(1)
        if func_name not in seen:
            functions.append(func_name)
            seen.add(func_name)

    return functions


def register_tvm_cuda_from_file(
    module: str,
    source: str | Path,
    include_dir: str | Path | None = None,
):
    """Compile a CUDA source and auto-register all TVM FFI entry points.

    Like :func:`register_tvm_cuda_kernels` but **automatically discovers**
    the entry-point function names by parsing the CUDA source code.  Any
    top-level ``void`` function whose parameter list contains
    ``tvm::ffi::TensorView`` is treated as a TVM FFI entry point and
    registered as ``"<module>.<function_name>"``.

    No explicit ``functions`` argument is required — the list is derived
    directly from the source, so adding or removing entry points in the
    ``.cu`` file is immediately reflected without touching the Python side.

    Parameters
    ----------
    module : str
        Module name under which the compiled kernels are registered.
        Must be unique within the process.
    source : str or Path
        Either a CUDA C/C++ source string, or a path to a ``.cu`` file
        (as a :class:`pathlib.Path` or a ``str`` pointing to an existing
        file).  When a path is given the file is read automatically::

            register_tvm_cuda_from_file('mykernels', Path(__file__).parent / 'mykernels.cu')

        A raw CUDA string is also accepted::

            register_tvm_cuda_from_file('mykernels', cuda_source_string)
    include_dir : str or Path or None, optional
        Directory path to add to the include search path during compilation.
        This allows ``#include`` directives in the source to find headers.

    Returns
    -------
    object
        The compiled CUDA module object.

    Raises
    ------
    TVMFFINotInstalledError
        If ``jax_tvm_ffi`` or ``tvm_ffi.cpp`` is not installed.
    TVMModuleAlreadyRegisteredError
        If *module* was already registered with a different source or
        a different set of discovered entry functions.
    ValueError
        If no TVM FFI entry functions are found in *source*.

    See Also
    --------
    register_tvm_cuda_kernels : Lower-level variant that requires an
        explicit ``functions`` list.

    Notes
    -----
    Entry-point detection uses two mechanisms (see
    :func:`_parse_tvm_entry_functions`):

    1. **Explicit functions** -- top-level ``void`` functions (at column 0)
       whose parameter lists contain ``tvm::ffi::TensorView``.

    2. **Annotation comments** -- lines matching ``// @tvm_ffi name``.
       This allows macro-generated FFI entry points to be registered::

           // @tvm_ffi binary_densemv_gather_auto_f32_bool
           FFI_GATHER_AUTO(_f32_bool, float, int8_t)

    Both mechanisms are applied; duplicates are removed automatically.
    """
    if not isinstance(module, str):
        raise TypeError(f"module must be a str, got {type(module).__name__}")
    if isinstance(source, Path):
        source = source.read_text()
    elif isinstance(source, str) and Path(source).is_file():
        source = Path(source).read_text()
    if not isinstance(source, str):
        raise TypeError(f"source must be a str or Path, got {type(source).__name__}")
    if not source.strip():
        raise ValueError(f"source for module '{module}' is empty.")

    functions = _parse_tvm_entry_functions(source)

    if not functions:
        raise ValueError(
            f"No TVM FFI entry functions found in the CUDA source for module '{module}'. "
            "Entry functions must be top-level 'void' functions (at column 0) "
            "with 'tvm::ffi::TensorView' parameters."
        )
    return register_tvm_cuda_kernels(
        source_code=source,
        module=module,
        functions=functions,
        include_dir=include_dir,
    )
