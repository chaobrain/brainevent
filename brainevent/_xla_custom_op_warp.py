# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import ctypes
import functools
import importlib.util
from packaging import version
from typing import Callable, Sequence, Dict, Union, NamedTuple

import jax
import numpy as np
from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from ._compatible_import import Primitive, register_custom_call, custom_call
from ._typing import KernelGenerator

__all__ = [
    'jaxtype_to_warptype',
    'jaxinfo_to_warpinfo',
    'warp_kernel',
]

# Holder for the custom callback to keep it alive.
_registered_warp_gpu_kernels = [None]
_registered_warp_gpu_kernel_to_id = {}

warp_installed = importlib.util.find_spec('warp') is not None
_warp_gpu_capsule = False

if warp_installed:
    import warp  # pylint: disable=import-error, import-outside-toplevel
    import warp.context  # pylint: disable=import-error, import-outside-toplevel
    import warp.types  # pylint: disable=import-error, import-outside-toplevel

    warp.config.enable_backward = False


class WarpKernel(NamedTuple):
    """
    A named tuple representing a compiled Warp kernel with configuration for GPU execution.

    This class encapsulates a Warp kernel along with its execution parameters, such as
    launch dimensions, tiling configuration, and memory aliasing information.

    Attributes
    ----------
    kernel : Callable
        The compiled Warp function that performs the actual computation on GPU.

    dim : Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]], optional
        The launch dimensions of the kernel. This can be:
        - An integer for 1D launch configuration
        - A sequence of integers for multi-dimensional launch
        - A callable that returns dimensions when invoked with kwargs
        If None, then 'tile' and 'block_dim' must be provided instead.

    tile : Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]], optional
        The tile dimensions for tile-based kernel operations. Only used when 'dim' is None.
        This can be an integer, sequence of integers, or a callable returning dimensions.
        See: https://nvidia.github.io/warp/modules/tiles.html

    block_dim : Union[int, Callable[..., int], None], optional
        The number of threads per block for kernel execution. Can be an integer or
        a callable that returns an integer when invoked with kwargs.
        Default is None, which uses 256 threads per block if not specified.

    input_output_aliases : Union[Dict[int, int], Callable[..., Dict[int, int]], None], optional
        A dictionary mapping output indices to input indices, indicating which
        output buffers can reuse the same memory as input buffers.
        This enables in-place operations to avoid unnecessary memory allocations.
        Can also be a callable that returns such a dictionary when invoked with kwargs.
    """
    kernel: Callable

    # "dim" describes the launch dimensions of the kernel.
    dim: Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]] = None

    # If "dim" is not provided, "tile" and "block_dim" should be provided.
    # Then, the kernel is launched with tile-based operation:
    #    https://nvidia.github.io/warp/modules/tiles.html
    tile: Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]] = None
    block_dim: Union[int, Callable[..., int], None] = None

    # input_output_aliases: Dict[int, int]. The input-output aliases.
    input_output_aliases: Union[Dict[int, int], Callable[..., Dict[int, int]], None] = None


def warp_kernel(
    fn: Callable = None,
    dim: Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]] = None,
    tile: Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]] = None,
    block_dim: Union[int, Callable[..., int], None] = None,
    input_output_aliases: Union[Dict[int, int], Callable[..., Dict[int, int]], None] = None
) -> Union[WarpKernel, Callable[[Callable], WarpKernel]]:
    """
    Creates a WarpKernel by compiling the provided function with Warp.

    This function can be used as a decorator or called directly to compile a Python
    function into an optimized WarpKernel for GPU execution. It supports configuring
    launch dimensions, tiling, block dimensions, and input-output aliases.

    Parameters
    ----------
    fn : Callable, optional
        The function to be compiled with Warp. If None, returns a partial function
        that can be used as a decorator.
    dim : Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]], optional
        The launch dimensions for the kernel. Can be an integer, sequence of integers,
        or a callable that returns dimensions when invoked with kwargs.
        If None, then 'tile' and 'block_dim' must be provided instead.
    tile : Union[int, Sequence[int], Callable[..., Sequence[int]], Callable[..., int]], optional
        The tile dimensions for tile-based kernel operations. Only used when 'dim' is None.
        Can be an integer, sequence of integers, or a callable returning dimensions.
    block_dim : Union[int, Callable[..., int], None], optional
        The number of threads per block for kernel execution. Can be an integer or
        a callable that returns an integer when invoked with kwargs.
    input_output_aliases : Union[Dict[int, int], Callable[..., Dict[int, int]], None], optional
        A dictionary mapping output indices to input indices, indicating which
        output buffers can reuse the same memory as input buffers.
        Can also be a callable that returns such a dictionary when invoked with kwargs.

    Returns
    -------
    Union[WarpKernel, Callable[..., WarpKernel]]
        If `fn` is provided, returns a WarpKernel instance containing the compiled function.
        If `fn` is None, returns a partial function that can be used as a decorator.

    Raises
    ------
    ImportError
        If Warp is not installed but is required to compile the GPU kernel.

    Examples
    --------
    # Direct function call
    >>> kernel = warp_kernel(my_function, dim=(16, 16))

    # As a decorator
    >>> @warp_kernel(block_dim=256)
    ... def my_function(x, y, out):
    ...     # function implementation
    ...     pass

    # With tile-based operation
    >>> @warp_kernel(tile=(32, 32), block_dim=128)
    ... def my_tiled_function(x, y, out):
    ...     # tiled implementation
    ...     pass
    """
    if fn is None:
        return functools.partial(
            warp_kernel,
            dim=dim,
            tile=tile,
            block_dim=block_dim,
            input_output_aliases=input_output_aliases
        )

    if not warp_installed:
        raise ImportError('Warp is required to compile the GPU kernel for the custom operator.')

    return WarpKernel(
        kernel=warp.kernel(fn),
        dim=dim,
        tile=tile,
        block_dim=block_dim,
        input_output_aliases=input_output_aliases
    )


def jaxtype_to_warptype(dtype):
    """
    Convert the JAX dtype to the Warp type.

    Args:
        dtype: np.dtype. The JAX dtype.

    Returns:
        ``Warp`` type.
    """
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
        raise ValueError(f"Unsupported dtype: {dtype}")


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
    shape = jax_info.shape
    return warp.array(dtype=dtype, ndim=len(shape))


def _shape_to_layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


def _warp_gpu_custom_callback(stream, buffers, opaque, opaque_len):
    # The descriptor is the form
    # <kernel-id>|<launch-dims>|<arg-dims-list>|<block-dim>
    # Example:  42|16,32|16,32;100;16,32|256
    kernel_id_str, dim_str, args_str, block_dim_str = opaque.decode().split("|")

    # Get the kernel from the registry.
    kernel_id = int(kernel_id_str)
    kernel = _registered_warp_gpu_kernels[kernel_id]

    # Parse launch dimensions.
    dims = [int(d) for d in dim_str.split(",")]
    bounds = warp.types.launch_bounds_t(dims)
    block_dim = int(block_dim_str)

    # Parse arguments.
    arg_strings = args_str.split(";")
    num_args = len(arg_strings)
    assert num_args == len(kernel.adj.args), "Incorrect number of arguments"

    # First param is the launch bounds.
    kernel_params = (ctypes.c_void_p * (1 + num_args))()
    kernel_params[0] = ctypes.addressof(bounds)

    # Parse array descriptors.
    args = []
    for i in range(num_args):
        dtype = kernel.adj.args[i].type.dtype
        shape = [int(d) for d in arg_strings[i].split(",")]
        strides = warp.types.strides_from_shape(shape, dtype)

        arr = warp.types.array_t(buffers[i], 0, len(shape), shape, strides)
        args.append(arr)  # keep a reference
        arg_ptr = ctypes.addressof(arr)

        kernel_params[i + 1] = arg_ptr

    # Get current device.
    device = warp.get_cuda_device(_get_jax_device().id)

    # Get kernel hooks.
    # Note: module was loaded during jit lowering.
    hooks = kernel.module.get_kernel_hooks(kernel, device)
    assert hooks.forward, "Failed to find kernel entry point"

    # Launch the kernel.
    warp_version = warp.__version__
    if version.parse(warp_version) >= version.parse("1.9.0"):
        warp_launch_kernel_func = warp.context.runtime.core.wp_cuda_launch_kernel
    else:
        warp_launch_kernel_func = warp.context.runtime.core.cuda_launch_kernel
    
    warp_launch_kernel_func(
        device.context,
        hooks.forward,
        bounds.size,
        0,  # max_blocks
        block_dim,  # threads_per_block
        hooks.forward_smem_bytes,
        kernel_params,
        stream
    )


# Create python-land custom call target.
warp_gpu_CCALL_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p,
    ctypes.c_size_t
)
warp_gpu_cc_callback = warp_gpu_CCALL_FUNC(_warp_gpu_custom_callback)
warp_gpu_ccall_address = ctypes.cast(warp_gpu_cc_callback, ctypes.c_void_p)

# def _warp_cpu_single_out_call(output_ptrs, input_ptrs):
#     kernel_id = int(kernel_id_str)
#     kernel = _registered_warp_gpu_kernels[kernel_id]
#
#     num_args = len(kernel.adj.args)
#
#     # First param is the launch bounds.
#     kernel_params = (ctypes.c_void_p * num_args)()
#
#     # Parse array descriptors.
#     args = []
#     for i in range(num_args):
#         dtype = kernel.adj.args[i].type.dtype
#         shape = [int(d) for d in arg_strings[i].split(",")]
#         strides = warp.types.strides_from_shape(shape, dtype)
#
#         arr = warp.types.array_t(input_ptrs[i], 0, len(shape), shape, strides)
#         args.append(arr)  # keep a reference
#         arg_ptr = ctypes.addressof(arr)
#         kernel_params[i] = arg_ptr
#
#     # compile the kernel #
#     # ------------------ #
#
#     # Get current device.
#     device = warp.device_from_jax(_get_jax_device())
#     # Get kernel hooks.
#     # Note: module was loaded during jit lowering.
#     hooks = kernel.module.get_kernel_hooks(kernel, device)
#     assert hooks.forward, "Failed to find kernel entry point"
#
#     # Launch the kernel.
#     hooks.forward(*kernel_params)


warp_cpu_CCALL_FUNC_single_out = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
)
# warp_cpu_callback_single_out = warp_cpu_CCALL_FUNC_single_out(_warp_cpu_single_out_call)
# warp_cpu_ccall_address_single_out = ctypes.cast(warp_cpu_callback_single_out, ctypes.c_void_p)


# def _warp_cpu_multiple_outs_call(output_ptrs, input_ptrs):
#     # The descriptor is the form
#     # <kernel-id>|<launch-dims>|<arg-dims-list>
#     # Example:  42|16,32|16,32;100;16,32
#     kernel_id_str, dim_str, args_str = opaque.decode().split("|")
#
#     # Get the kernel from the registry.
#     kernel_id = int(kernel_id_str)
#     kernel = _registered_warp_gpu_kernels[kernel_id]
#
#     # Parse launch dimensions.
#     dims = [int(d) for d in dim_str.split(",")]
#     bounds = warp.types.launch_bounds_t(dims)
#
#     # Parse arguments.
#     arg_strings = args_str.split(";")
#     num_args = len(arg_strings)
#     assert num_args == len(kernel.adj.args), "Incorrect number of arguments"
#
#     # First param is the launch bounds.
#     kernel_params = (ctypes.c_void_p * (1 + num_args))()
#     kernel_params[0] = ctypes.addressof(bounds)
#
#     # Parse array descriptors.
#     args = []
#     for i in range(num_args):
#         dtype = kernel.adj.args[i].type.dtype
#         shape = [int(d) for d in arg_strings[i].split(",")]
#         strides = warp.types.strides_from_shape(shape, dtype)
#
#         arr = warp.types.array_t(buffers[i], 0, len(shape), shape, strides)
#         args.append(arr)  # keep a reference
#         arg_ptr = ctypes.addressof(arr)
#
#         kernel_params[i + 1] = arg_ptr
#
#     # Get current device.
#     device = warp.device_from_jax(_get_jax_device())
#
#     # Get kernel hooks.
#     # Note: module was loaded during jit lowering.
#     hooks = kernel.module.get_kernel_hooks(kernel, device)
#     assert hooks.forward, "Failed to find kernel entry point"
#
#     # Launch the kernel.
#     warp.context.runtime.core.cuda_launch_kernel(
#         device.context,
#         hooks.forward,
#         bounds.size,
#         0,  # max_blocks
#         256,  # threads_per_block
#         hooks.forward_smem_bytes,
#         kernel_params,
#         stream
#     )


warp_cpu_CCALL_FUNC_multi_outs = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
)


# warp_cpu_callback_multi_out = warp_cpu_CCALL_FUNC_multi_outs(_warp_cpu_multiple_outs_call)
# warp_cpu_ccall_address_multi_out = ctypes.cast(warp_cpu_callback_multi_out, ctypes.c_void_p)


def _warp_gpu_register_capsule():
    global _warp_gpu_capsule
    if _warp_gpu_capsule:
        return

    _warp_gpu_capsule = True

    # Put the custom call into a capsule, as required by XLA.
    warp_PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    warp_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    warp_PyCapsule_New.restype = ctypes.py_object
    warp_PyCapsule_New.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        warp_PyCapsule_Destructor
    )
    warp_capsule = warp_PyCapsule_New(
        warp_gpu_ccall_address.value,
        b"xla._CUSTOM_CALL_TARGET",
        warp_PyCapsule_Destructor(0)
    )

    # Register the callback in XLA.
    register_custom_call("brainevent_warp_gpu_call", warp_capsule, "gpu")


def _register_warp_kernel(wp_kernel) -> int:
    if wp_kernel not in _registered_warp_gpu_kernel_to_id:
        id_ = len(_registered_warp_gpu_kernels)
        _registered_warp_gpu_kernels.append(wp_kernel)
        _registered_warp_gpu_kernel_to_id[wp_kernel] = id_
    else:
        id_ = _registered_warp_gpu_kernel_to_id[wp_kernel]
    return id_


def _warp_get_vecmat_shape(warp_type):
    if hasattr(warp_type, 'dtype'):
        if hasattr(warp_type.dtype, "_shape_"):
            return warp_type.dtype._shape_
    return []


def _warp_strip_vecmat_dimensions(warp_arg, actual_shape):
    shape = _warp_get_vecmat_shape(warp_arg.type)
    for i, s in enumerate(reversed(shape)):
        item = actual_shape[-i - 1]
        if s != item:
            raise Exception(f"The vector/matrix shape for argument {warp_arg.label} does not match")
    return actual_shape[: len(actual_shape) - len(shape)]


def _warp_collapse_into_leading_dimension(warp_arg, actual_shape):
    if len(actual_shape) < warp_arg.type.ndim:
        raise Exception(f"Argument {warp_arg.label} has too few non-matrix/vector dimensions")
    index_rest = len(actual_shape) - warp_arg.type.ndim + 1
    leading_size = functools.reduce(lambda x, y: x * y, actual_shape[:index_rest])
    return [leading_size] + actual_shape[index_rest:]


# Infer array dimensions from input type.
def _warp_infer_dimensions(warp_arg, actual_shape):
    actual_shape = _warp_strip_vecmat_dimensions(warp_arg, actual_shape)
    return _warp_collapse_into_leading_dimension(warp_arg, actual_shape)


def _get_jax_device():
    # check if jax.default_device() context manager is active
    device = jax.config.jax_default_device
    # if default device is not set, use first device
    if device is None:
        device = jax.local_devices()[0]
    return device


def _warp_base_type_is_compatible(warp_type, jax_ir_type):
    jax_ir_to_warp = {
        "f16": warp.float16,
        "f32": warp.float32,
        "f64": warp.float64,
        "i8": warp.int8,
        "i16": warp.int16,
        "i32": warp.int32,
        "i64": warp.int64,
        "ui8": warp.uint8,
        "ui16": warp.uint16,
        "ui32": warp.uint32,
        "ui64": warp.uint64,
        "b1": warp.bool,
        "i1": warp.bool,
    }
    expected_warp_type = jax_ir_to_warp.get(str(jax_ir_type))
    if expected_warp_type is not None:
        if hasattr(warp_type, "_wp_scalar_type_"):
            return warp_type._wp_scalar_type_ == expected_warp_type
        else:
            return warp_type == expected_warp_type
    else:
        raise TypeError(f"Invalid or unsupported data type: {jax_ir_type}")


def _warp_gpu_lowering(
    kernel_generator: KernelGenerator,
    ctx,
    *args,
    **kwargs,
):
    if not warp_installed:
        raise ImportError('Warp is required to compile the GPU kernel for the custom operator.')
    _warp_gpu_register_capsule()

    # ------------------
    # kernels
    # ------------------
    wp_kernel: WarpKernel = kernel_generator(**kwargs)
    assert isinstance(wp_kernel.kernel, warp.context.Kernel), (
        f'The kernel should be a Warp '
        f'kernel. But we got {wp_kernel}'
    )

    kernel_id = _register_warp_kernel(wp_kernel.kernel)

    # ------------------
    # block dimensions
    # ------------------
    block_dim = wp_kernel.block_dim
    if callable(block_dim):
        block_dim = block_dim(**kwargs)
    if isinstance(block_dim, int):
        pass
    elif block_dim is None:
        block_dim = 256
    else:
        raise ValueError(
            f"Invalid block dimensions, expected "
            f"int, got {block_dim}"
        )

    # ------------------
    # launch dimensions
    # ------------------
    warp_dims = wp_kernel.dim
    if warp_dims is None:
        assert wp_kernel.tile is not None, ('The tile dimensions should be provided when '
                                            'the launch dimensions are not provided.')
        assert wp_kernel.block_dim is not None, (
            'The block dimensions should be provided when the tile dimensions are provided.'
        )
        warp_dims = wp_kernel.tile
        if callable(warp_dims):
            warp_dims = warp_dims(**kwargs)
        if isinstance(warp_dims, int):
            warp_dims = (warp_dims,)
        assert isinstance(warp_dims, (tuple, list)), (
            f"Invalid launch dimensions, expected "
            f"tuple or list, got {warp_dims}"
        )
        warp_dims = tuple(warp_dims) + (block_dim,)
    else:
        if callable(warp_dims):
            warp_dims = warp_dims(**kwargs)
        if isinstance(warp_dims, int):
            warp_dims = (warp_dims,)
        assert isinstance(warp_dims, (tuple, list)), (
            f"Invalid launch dimensions, expected "
            f"tuple or list, got {warp_dims}"
        )
        warp_dims = tuple(warp_dims)

    # TODO: This may not be necessary, but it is perhaps better not to be
    #       mucking with kernel loading while already running the workload.
    module = wp_kernel.kernel.module
    device = warp.device_from_jax(_get_jax_device())
    if not module.load(device, block_dim):
        raise Exception("Could not load kernel on device")

    # ------
    # inputs
    # ------
    # Figure out the types and shapes of the input arrays.
    arg_strings = []
    operand_layouts = []
    for actual, warg in zip(args, wp_kernel.kernel.adj.args):
        rtt = ir.RankedTensorType(actual.type)
        _warp_strip_vecmat_dimensions(warg, rtt.shape)
        if hasattr(warg.type, 'ndim'):
            if len(rtt.shape) < warg.type.ndim:
                raise Exception(f"Argument {warg.label} has too few non-matrix/vector dimensions")
        arg_strings.append(",".join([str(d) for d in rtt.shape]))
        operand_layouts.append(_shape_to_layout(rtt.shape))

    # ------------------
    # output information
    # ------------------
    # Figure out the types and shapes of the output arrays.
    outs = ctx.avals_out
    result_layouts, result_types = [], []
    for out in outs:
        arg_strings.append(",".join([str(d) for d in out.shape]))
        result_layouts.append(_shape_to_layout(out.shape))
        result_types.append(mlir.aval_to_ir_type(out))

    # Build opaque descriptor for callback.
    dims_str = ",".join([str(d) for d in warp_dims])
    args_str = ";".join(arg_strings)
    descriptor = f"{kernel_id}|{dims_str}|{args_str}|{block_dim}"

    # ---------------------
    # input_output_aliases
    # ---------------------

    input_output_aliases = wp_kernel.input_output_aliases
    if callable(input_output_aliases):
        input_output_aliases = input_output_aliases(**kwargs)

    # custom call
    out = custom_call(
        b"brainevent_warp_gpu_call",
        result_types=result_types,
        operands=args,
        backend_config=descriptor.encode("utf-8"),
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        operand_output_aliases=input_output_aliases,
    ).results
    return out


def register_warp_gpu_translation(
    primitive: Primitive,
    kernel_generator: KernelGenerator,
):
    """
    Register the Warp GPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        kernel_generator: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the Warp kernel.
    """
    # register the lowering rule
    mlir.register_lowering(
        primitive,
        functools.partial(_warp_gpu_lowering, kernel_generator),
        platform="gpu",
    )
