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
import importlib.util
from typing import Protocol, Union, Tuple, Sequence

import jax
import numpy as np
from jax import tree_util
from jax.interpreters import ad

from brainevent._compatible_import import Primitive

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
    'defjvp',
    'general_batching_rule',
    'jaxtype_to_warptype',
    'jaxinfo_to_warpinfo',
    'check_pallas_jax_version',
]

_MIN_JAX_VERSION_FOR_PALLAS = (0, 7, 1)


def check_pallas_jax_version():
    """Check that JAX version is >= 0.7.1 for Pallas kernel support.

    Raises:
        RuntimeError: If the installed JAX version is older than 0.7.1.
    """
    if jax.__version_info__ < _MIN_JAX_VERSION_FOR_PALLAS:
        min_ver = '.'.join(str(v) for v in _MIN_JAX_VERSION_FOR_PALLAS)
        raise RuntimeError(
            f"Pallas kernels require JAX >= {min_ver}, "
            f"but found JAX {jax.__version__}. "
            f"Please upgrade JAX: pip install --upgrade jax"
        )


def register_tvm_cuda_kernels(
    source_code: str,
    module: str,
    functions: Sequence[str],
):
    """Compile CUDA kernels and register with JAX FFI."""

    if not tvmffi_installed:
        return

    if not isinstance(source_code, str):
        return ValueError("source_code must be a string")
    if not isinstance(module, str):
        return ValueError("module must be a string")
    if not isinstance(functions, Sequence) or not all(isinstance(f, str) for f in functions):
        return ValueError("functions must be a sequence of strings")

    # Compile CUDA module
    _cuda_module = tvm_ffi.cpp.load_inline(
        name=module,
        cuda_sources=source_code,
        functions=functions,
    )

    # Register each kernel with JAX FFI
    for name in functions:
        jax_tvm_ffi.register_ffi_target(
            f"{module}.{name}",
            getattr(_cuda_module, name),
            ["args", "rets", "ctx.stream"],
            platform="gpu",
        )


def defjvp(primitive, *jvp_rules):
    """
    Define JVP rules for any JAX primitive.

    This function allows defining Jacobian-vector product (JVP) rules for JAX primitives,
    extending the functionality of ``jax.interpreters.ad.defjvp``. While the standard
    JAX function primarily supports primitives that return a single result
    (``multiple_results=False``), this implementation supports defining independent
    JVP rules for each input parameter regardless of whether the primitive returns
    single or multiple results.

    This is particularly useful for custom operations or primitives where different
    inputs might have different differentiation rules or where the primitive naturally
    produces multiple outputs that need distinct handling in automatic differentiation.

    For concrete usage examples, refer to the test file ``test_ad_support.py``.

    Args:
        primitive: The JAX ``Primitive`` object or an ``XLACustomKernel`` instance
            for which the JVP rule is being defined. If an ``XLACustomKernel`` is
            provided, its underlying ``Primitive`` is extracted.
        *jvp_rules: A variable number of functions, each representing the JVP rule
            corresponding to a primal input argument of the primitive. Each rule
            function should accept the tangent vector for its corresponding primal input,
            followed by all the primal inputs, and any keyword arguments passed to the
            primitive. It should return the tangent vector(s) corresponding to the
            primitive's output(s). If a rule is ``None``, it implies the JVP for that
            input is zero.
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
    return tree_util.tree_map(ad.add_tangents, xs, ys, is_leaf=lambda a: isinstance(a, ad.Zero))


def general_batching_rule(prim, args, axes, **kwargs):
    """
    Implements a general batching rule for JAX primitive operations.

    This function handles batching for JAX primitives by separating batched and non-batched
    arguments, then applying the primitive to each element in the batch using jax.lax.scan.

    Args:
        prim: The JAX primitive operation to be batched.
        args: Sequence of input arguments to the primitive.
        axes: Sequence of axis indices indicating the batch dimension for each argument.
              None indicates that the corresponding argument is not batched.
        **kwargs: Additional keyword arguments to pass to the primitive.

    Returns:
        tuple: A tuple containing:
            - outs: The batched outputs from applying the primitive.
            - out_dim: A pytree with the same structure as outs, indicating
              the batch dimensions of each output.

    Note:
        This function moves all batch dimensions to the leading axis (0) before
        applying scan, then processes each slice of the batched inputs.
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
    """A protocol defining objects that have `shape` and `dtype` attributes.

    This protocol is used for type hinting to indicate that an object is expected
    to provide information about its tensor shape (as a tuple of integers) and
    its data type (as a NumPy dtype). It's commonly used in JAX and related
    libraries to specify the expected structure of abstract arrays or outputs
    without requiring a specific concrete class like `jax.core.ShapedArray`.

    Examples:

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
        """The data type of the tensor elements (e.g., np.float32)."""
        ...


OutType = Union[ShapeDtype, Sequence[ShapeDtype]]


def _transform_to_shapedarray(a):
    return jax.core.ShapedArray(a.shape, a.dtype)


def abstract_arguments(outs):
    outs = jax.tree.map(_transform_to_shapedarray, outs)
    outs, tree_def = jax.tree.flatten(outs)
    return outs, tree_def


def jaxtype_to_warptype(dtype):
    """
    Convert the JAX dtype to the Warp type.

    Args:
        dtype: np.dtype. The JAX dtype.

    Returns:
        ``Warp`` type.
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
    """
    Convert JAX shape and dtype information to a compatible Warp array type.

    This function takes a JAX ShapeDtypeStruct object and creates an appropriate
    Warp array type with the corresponding data type and dimensionality.
    This is useful when interfacing between JAX and Warp, allowing JAX arrays
    to be processed by Warp kernels.

    Parameters
    ----------
    jax_info : jax.ShapeDtypeStruct
        A JAX structure containing shape and dtype information for an array.

    Returns
    -------
    warp.types.array
        A Warp array type with matching data type and dimensionality that can be
        used in Warp kernel definitions.

    Examples
    --------
    >>> array_info = jax.ShapeDtypeStruct(shape=(32, 32), dtype=np.float32)
    >>> warp_info = jaxinfo_to_warpinfo(array_info)
    >>> # Use warp_info in kernel definition

    See Also
    --------
    dtype_to_warp_type : Function to convert numpy/JAX dtypes to Warp types.
    """
    dtype = jaxtype_to_warptype(jax_info.dtype)
    return warp.array(dtype=dtype, ndim=jax_info.ndim)
