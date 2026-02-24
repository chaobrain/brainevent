# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import functools
import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Protocol, Union, Tuple, Sequence

import jax
import numpy as np
from jax import tree_util
from jax.interpreters import ad

from brainevent._compatible_import import Primitive
from brainevent._error import TVMFFINotInstalledError, TVMModuleAlreadyRegisteredError

warp_installed = importlib.util.find_spec('warp') is not None
tvmffi_installed = importlib.util.find_spec('jax_tvm_ffi') is not None

# Try to import TVM FFI - will fail gracefully if not available
if tvmffi_installed:
    try:
        import jax_tvm_ffi
        import tvm_ffi.cpp
    except:
        tvmffi_installed = False

# Try to import Warp - will fail gracefully if not available
if warp_installed:
    try:
        import warp  # pylint: disable=import-error, import-outside-toplevel

        warp.config.quiet = True
    except:
        warp_installed = False

__all__ = [
    'register_tvm_cuda_kernels',
    'register_tvm_cuda_from_file',
    'defjvp',
    'general_batching_rule',
    'jaxtype_to_warptype',
    'jaxinfo_to_warpinfo',
    'check_pallas_jax_version',
    'check_warp_installed',
]

_MIN_JAX_VERSION_FOR_PALLAS = (0, 7, 1)

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


def check_pallas_jax_version():
    """Check that the installed JAX version satisfies Pallas requirements.

    Pallas kernels require JAX >= 0.7.1.  This function verifies the
    installed version and raises an informative error if it is too old.

    Raises
    ------
    RuntimeError
        If the installed JAX version is older than 0.7.1.

    Examples
    --------
    .. code-block:: python

        >>> check_pallas_jax_version()  # succeeds silently on JAX >= 0.7.1
    """
    if jax.__version_info__ < _MIN_JAX_VERSION_FOR_PALLAS:
        min_ver = '.'.join(str(v) for v in _MIN_JAX_VERSION_FOR_PALLAS)
        raise RuntimeError(
            f"Pallas kernels require JAX >= {min_ver}, "
            f"but found JAX {jax.__version__}. "
            f"Please upgrade JAX: pip install --upgrade jax"
        )


def check_warp_installed():
    """Check that the Warp package is installed and importable.

    This function verifies that the ``warp`` Python package is available
    at runtime.  It is called automatically before dispatching to a Warp
    kernel backend.

    Raises
    ------
    RuntimeError
        If the ``warp`` package is not installed or could not be
        imported.

    See Also
    --------
    jaxtype_to_warptype : Convert a JAX/NumPy dtype to a Warp type.
    jaxinfo_to_warpinfo : Convert a ``jax.ShapeDtypeStruct`` to a Warp
        array type.

    Examples
    --------
    .. code-block:: python

        >>> check_warp_installed()  # succeeds silently when warp is installed
    """
    if not warp_installed:
        raise RuntimeError(
            "Warp kernels require the 'warp' package, but it is not installed.\n"
            "Please install Warp using one of the following methods:\n"
            "  pip install warp-lang\n"
            "For more information, visit: https://nvidia.github.io/warp/user_guide/installation.html"
        )


def register_tvm_cuda_kernels(
    source_code: str,
    module: str,
    functions: Sequence[str],
    arg_spec: Sequence[str] = ("args", "rets", "ctx.stream"),
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

    # Compile CUDA module
    _cuda_module = tvm_ffi.cpp.load_inline(name=module, cuda_sources=source_code, functions=functions)

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
    return register_tvm_cuda_kernels(source_code=source, module=module, functions=functions)


def defjvp(primitive, *jvp_rules):
    """Define per-input JVP rules for a JAX primitive.

    This function allows defining Jacobian-vector product (JVP) rules
    for JAX primitives, extending the functionality of
    ``jax.interpreters.ad.defjvp``.  While the standard JAX function
    primarily supports primitives that return a single result
    (``multiple_results=False``), this implementation supports defining
    independent JVP rules for each input parameter regardless of whether
    the primitive returns single or multiple results.

    This is particularly useful for custom operations or primitives
    where different inputs have different differentiation rules, or where
    the primitive produces multiple outputs that need distinct handling
    in automatic differentiation.

    Parameters
    ----------
    primitive : Primitive or XLACustomKernel
        The JAX ``Primitive`` object (or an ``XLACustomKernel`` instance,
        from which the underlying ``Primitive`` is extracted) for which
        the JVP rule is being defined.
    *jvp_rules : callable or None
        One callable per input primal.  Each callable has the signature
        ``rule(tangent, *primals, **params) -> tangent_out`` and computes
        the tangent contribution from the corresponding input.  Pass
        ``None`` for inputs whose JVP contribution is zero.

    Raises
    ------
    AssertionError
        If *primitive* (after unwrapping) is not a ``Primitive`` instance.

    See Also
    --------
    XLACustomKernel.def_jvp_rule : Register a single monolithic JVP
        rule for an ``XLACustomKernel``.
    XLACustomKernel.def_jvp_rule2 : Convenience wrapper around this
        function.
    general_batching_rule : General batching rule for custom primitives.

    Notes
    -----
    When the primitive has ``multiple_results=True``, a custom internal
    JVP implementation (``_standard_jvp``) is used that correctly
    handles tuple outputs.  For single-result primitives, the standard
    ``jax.interpreters.ad.standard_jvp`` is used.

    Examples
    --------
    .. code-block:: python

        >>> # Assume `my_prim` is a JAX Primitive with two inputs.
        >>> def jvp_rule_input0(tangent, x, y, **kw):
        ...     return tangent * y
        >>> def jvp_rule_input1(tangent, x, y, **kw):
        ...     return tangent * x
        >>> defjvp(my_prim, jvp_rule_input0, jvp_rule_input1)
    """
    # Import XLACustomKernel locally to avoid circular dependencies.
    from .main import XLACustomKernel

    # If the input is an XLACustomKernel, extract the underlying JAX primitive.
    if isinstance(primitive, XLACustomKernel):
        primitive = primitive.primitive
    # Ensure that the 'primitive' argument is indeed a JAX Primitive object.
    assert isinstance(primitive, Primitive), f'The primitive should be a JAX primitive. But we got {primitive}'

    # Check if the primitive returns multiple results.
    if primitive.multiple_results:
        # If yes, use the custom _standard_jvp function designed to handle multiple results.
        # ad.primitive_jvps is the JAX registry for JVP rules.
        # functools.partial pre-fills the jvp_rules and primitive arguments for _standard_jvp.
        ad.primitive_jvps[primitive] = functools.partial(_standard_jvp, jvp_rules, primitive)
    else:
        # If no (single result), use the standard JAX JVP handler (ad.standard_jvp).
        # This maintains compatibility with standard JAX behavior for single-result primitives.
        ad.primitive_jvps[primitive] = functools.partial(ad.standard_jvp, jvp_rules, primitive)


def _standard_jvp(jvp_rules, primitive: Primitive, primals, tangents, **params):
    """Compute the JVP for a multi-result primitive.

    This is an internal helper used by :func:`defjvp` when the primitive
    has ``multiple_results=True``.  It binds the primals, evaluates
    each per-input JVP rule whose tangent is not ``Zero``, and sums the
    resulting tangent contributions.

    Parameters
    ----------
    jvp_rules : sequence of callable or None
        Per-input JVP rules (see :func:`defjvp`).
    primitive : Primitive
        The JAX primitive.
    primals : tuple
        Primal input values.
    tangents : tuple
        Tangent input values (may contain ``ad.Zero``).
    **params
        Additional keyword arguments forwarded from the primitive bind.

    Returns
    -------
    tuple
        A pair ``(val_out, tangents_out)`` where *val_out* is the tuple
        of primal outputs and *tangents_out* is the summed tangent
        contributions with the same pytree structure as *val_out*.
    """
    assert primitive.multiple_results
    val_out = tuple(primitive.bind(*primals, **params))
    tree = tree_util.tree_structure(val_out)
    tangents_out = []
    for rule, t in zip(jvp_rules, tangents):
        if rule is not None and type(t) is not ad.Zero:
            r = tuple(rule(t, *primals, **params))
            tangents_out.append(r)
            assert tree_util.tree_structure(r) == tree
    r = functools.reduce(
        _add_tangents,
        tangents_out,
        tree_util.tree_map(
            # compatible with JAX 0.4.34
            lambda a: (
                ad.Zero.from_primal_value(a)
                if jax.__version_info__ >= (0, 4, 34) else
                ad.Zero.from_value(a)
            ),
            val_out
        )
    )
    return val_out, r


def _add_tangents(xs, ys):
    """Element-wise addition of two tangent pytrees.

    Parameters
    ----------
    xs : pytree
        First tangent pytree (may contain ``ad.Zero`` leaves).
    ys : pytree
        Second tangent pytree (may contain ``ad.Zero`` leaves).

    Returns
    -------
    pytree
        The element-wise sum of *xs* and *ys*.
    """
    return tree_util.tree_map(ad.add_tangents, xs, ys, is_leaf=lambda a: isinstance(a, ad.Zero))


def general_batching_rule(prim, args, axes, **kwargs):
    """General-purpose batching rule for custom JAX primitives.

    Implements batching by separating batched and non-batched arguments,
    moving all batch dimensions to axis 0, and then applying the primitive
    to each element in the batch via ``jax.lax.scan``.

    This function is registered as the default batching rule for every
    :class:`XLACustomKernel` during initialization.

    Parameters
    ----------
    prim : Primitive
        The JAX primitive operation to be batched.
    args : sequence of array_like
        Input arguments to the primitive.
    axes : sequence of int or None
        Batch dimension index for each argument.  ``None`` indicates
        that the corresponding argument is not batched.
    **kwargs
        Additional keyword arguments forwarded to the primitive.

    Returns
    -------
    outs : pytree
        The batched outputs from applying the primitive.
    out_dim : pytree
        A pytree with the same structure as *outs*, where every leaf
        is ``0``, indicating that the batch dimension is the leading
        axis of each output.

    Notes
    -----
    All batch dimensions are moved to axis 0 before scanning.  The scan
    carry is unused (always ``0``); only the stacked scan outputs are
    returned.

    See Also
    --------
    XLACustomKernel.register_general_batching : Registers this function
        as the batching rule for a primitive.
    XLACustomKernel.def_batching_rule : Override with a custom batching
        rule.

    Examples
    --------
    .. code-block:: python

        >>> import functools
        >>> from jax.interpreters import batching
        >>> batching.primitive_batchers[my_prim] = functools.partial(
        ...     general_batching_rule, my_prim
        ... )
    """
    batch_axes, batch_args, non_batch_args = [], {}, {}
    for ax_i, ax in enumerate(axes):
        if ax is None:
            non_batch_args[f'ax{ax_i}'] = args[ax_i]
        else:
            batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jax.numpy.moveaxis(args[ax_i], ax, 0)
            batch_axes.append(ax_i)

    def f(_, x):
        """
        Internal function for jax.lax.scan that applies the primitive to a single batch element.

        Args:
            _: Carry value (unused).
            x: Dictionary containing the current batch slice for each batched argument.

        Returns:
            tuple: (carry value, primitive output)
        """
        pars = tuple(
            [(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
             for i in range(len(axes))]
        )
        return 0, prim(*pars, **kwargs)

    _, outs = jax.lax.scan(f, 0, batch_args)
    out_vals, out_tree = jax.tree.flatten(outs)
    out_dim = jax.tree.unflatten(out_tree, (0,) * len(out_vals))
    return outs, out_dim


class ShapeDtype(Protocol):
    """Protocol for objects that expose ``shape`` and ``dtype`` attributes.

    This protocol is used for type hinting to indicate that an object
    provides information about its tensor shape (as a tuple of integers)
    and its data type (as a NumPy dtype).  It is commonly satisfied by
    ``jax.ShapeDtypeStruct``, ``jax.core.ShapedArray``, and any
    user-defined class with the matching attributes.

    See Also
    --------
    OutType : Type alias combining ``ShapeDtype`` and sequences thereof.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from typing import Tuple
        >>>
        >>> class MyTensorSpec:
        ...     def __init__(self, shape: Tuple[int, ...], dtype: np.dtype):
        ...         self._shape = shape
        ...         self._dtype = dtype
        ...
        ...     @property
        ...     def shape(self) -> Tuple[int, ...]:
        ...         return self._shape
        ...
        ...     @property
        ...     def dtype(self) -> np.dtype:
        ...         return self._dtype
        >>>
        >>> def process_spec(spec: ShapeDtype):
        ...     print(f"Shape: {spec.shape}, Dtype: {spec.dtype}")
        >>>
        >>> spec = MyTensorSpec(shape=(10, 20), dtype=np.float32)
        >>> process_spec(spec)
        Shape: (10, 20), Dtype: float32
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the tensor as a tuple of integers."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """The data type of the tensor elements (e.g., ``np.float32``)."""
        ...


OutType = Union[ShapeDtype, Sequence[ShapeDtype]]


def _transform_to_shapedarray(a):
    """Convert an object with ``shape`` and ``dtype`` to a ``jax.core.ShapedArray``.

    Parameters
    ----------
    a : ShapeDtype
        Any object satisfying the :class:`ShapeDtype` protocol.

    Returns
    -------
    jax.core.ShapedArray
        The corresponding JAX abstract array.
    """
    return jax.core.ShapedArray(a.shape, a.dtype)


def abstract_arguments(outs):
    """Flatten an output specification into abstract ``ShapedArray`` objects.

    Takes a pytree of objects with ``shape`` and ``dtype`` attributes
    (e.g., ``jax.ShapeDtypeStruct``) and returns a flat list of
    ``jax.core.ShapedArray`` instances together with the tree definition
    needed to reconstruct the original structure.

    Parameters
    ----------
    outs : OutType
        An output specification -- a single ``ShapeDtype`` object or a
        pytree (list, tuple, dict, ...) of them.

    Returns
    -------
    flat_outs : list of jax.core.ShapedArray
        Flattened list of abstract arrays.
    tree_def : jax.tree_util.PyTreeDef
        The pytree definition that can be used to unflatten the outputs
        back into the original structure.

    See Also
    --------
    ShapeDtype : Protocol describing the expected input objects.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> spec = [
        ...     jax.ShapeDtypeStruct((10,), jnp.float32),
        ...     jax.ShapeDtypeStruct((5, 3), jnp.int32),
        ... ]
        >>> flat, treedef = abstract_arguments(spec)
        >>> len(flat)
        2
        >>> flat[0].shape
        (10,)
    """
    outs = jax.tree.map(_transform_to_shapedarray, outs)
    outs, tree_def = jax.tree.flatten(outs)
    return outs, tree_def


def jaxtype_to_warptype(dtype):
    """Convert a JAX / NumPy dtype to the corresponding Warp scalar type.

    Maps standard NumPy data types (which are also used by JAX) to their
    Warp equivalents.  This is needed when constructing Warp kernel
    signatures or Warp array types from JAX metadata.

    Parameters
    ----------
    dtype : numpy.dtype or type
        The data type to convert.  Accepts any object that can be
        compared with NumPy scalar types (e.g., ``np.float32``,
        ``jnp.float32``, ``np.dtype('float32')``).

    Returns
    -------
    warp type
        The corresponding Warp scalar type (e.g., ``warp.float32``,
        ``warp.int32``, ``warp.bool``).

    Raises
    ------
    ImportError
        If the ``warp`` package is not installed.
    ValueError
        If *dtype* does not correspond to any supported Warp type.
        Supported types include: ``float16``, ``float32``, ``float64``,
        ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``,
        ``uint16``, ``uint32``, ``uint64``, and ``bool_``.

    See Also
    --------
    jaxinfo_to_warpinfo : Convert a full ``jax.ShapeDtypeStruct`` to a
        Warp array type.
    check_warp_installed : Verify that Warp is available.

    Notes
    -----
    The mapping covers all scalar types supported by both NumPy and
    Warp: ``float16``, ``float32``, ``float64``, ``int8`` through
    ``int64``, ``uint8`` through ``uint64``, and ``bool_``.  Complex
    types are not supported by Warp and will raise ``ValueError``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> warp_type = jaxtype_to_warptype(np.float32)
        >>> warp_type  # warp.float32
    """
    if not warp_installed:
        raise ImportError('Warp is required to convert JAX dtypes to Warp types.')

    # float
    if dtype == np.float16:
        return warp.float16
    elif dtype == np.float32:
        return warp.float32
    elif dtype == np.float64:
        return warp.float64

    # integer
    elif dtype == np.int8:
        return warp.int8
    elif dtype == np.int16:
        return warp.int16
    elif dtype == np.int32:
        return warp.int32
    elif dtype == np.int64:
        return warp.int64

    # unsigned integer
    elif dtype == np.uint8:
        return warp.uint8
    elif dtype == np.uint16:
        return warp.uint16
    elif dtype == np.uint32:
        return warp.uint32
    elif dtype == np.uint64:
        return warp.uint64

    # boolean
    elif dtype == np.bool_:
        return warp.bool
    else:
        raise ValueError(f"Warp does not support computations with dtype: {dtype}")


def jaxinfo_to_warpinfo(jax_info: jax.ShapeDtypeStruct):
    """Convert a ``jax.ShapeDtypeStruct`` to a Warp array type descriptor.

    Takes a JAX shape-and-dtype specification and creates the
    corresponding Warp array type with matching data type and
    dimensionality.  This is useful when building Warp kernel
    signatures from JAX output specifications.

    Parameters
    ----------
    jax_info : jax.ShapeDtypeStruct
        A JAX structure containing ``shape``, ``dtype``, and ``ndim``
        attributes describing an array.

    Returns
    -------
    warp.types.array
        A Warp array type with matching data type and number of
        dimensions, suitable for use in Warp kernel definitions.

    Raises
    ------
    ImportError
        If the ``warp`` package is not installed (propagated from
        :func:`jaxtype_to_warptype`).
    ValueError
        If the dtype in *jax_info* is not supported by Warp (propagated
        from :func:`jaxtype_to_warptype`).

    See Also
    --------
    jaxtype_to_warptype : Convert a single dtype to a Warp type.
    check_warp_installed : Verify that Warp is available.

    Notes
    -----
    The resulting Warp array type is constructed via
    ``warp.array(dtype=..., ndim=...)`` which creates a Warp type
    descriptor (not an actual array).  This is typically used in Warp
    kernel function signatures to define input/output types.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import numpy as np
        >>> info = jax.ShapeDtypeStruct(shape=(32, 32), dtype=np.float32)
        >>> warp_arr_type = jaxinfo_to_warpinfo(info)
    """
    dtype = jaxtype_to_warptype(jax_info.dtype)
    return warp.array(dtype=dtype, ndim=jax_info.ndim)
