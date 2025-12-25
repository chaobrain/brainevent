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
from typing import Callable, Sequence, Dict, Union, NamedTuple

import jax
import numpy as np
from jax.interpreters import mlir

from brainevent._compatible_import import Primitive
from brainevent._typing import KernelGenerator
from .warp_customcall import _custom_call_gpu_lowering
from .warp_ffi import _ffi_gpu_lowering

__all__ = [
    'warp_kernel',
    'jaxinfo_to_warpinfo',
    'jaxtype_to_warptype',
]

warp_installed = importlib.util.find_spec('warp') is not None

if warp_installed:
    import warp  # pylint: disable=import-error, import-outside-toplevel


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

    vmap_method: str = 'sequential'
    module_preload_mode: str = 'CURRENT_DEVICE'


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
            input_output_aliases=input_output_aliases,
        )

    if not warp_installed:
        raise ImportError('Warp is required to compile the GPU kernel for the custom operator.')

    return WarpKernel(
        kernel=warp.kernel(fn),
        dim=dim,
        tile=tile,
        block_dim=block_dim,
        input_output_aliases=input_output_aliases,
    )


def register_warp_gpu_translation(
    primitive: Primitive,
    kernel_generator: KernelGenerator,
    version: str = 'custom_call',
):
    if version == 'ffi':
        rule = functools.partial(_ffi_gpu_lowering, kernel_generator)
    elif version == 'custom_call':
        rule = functools.partial(_custom_call_gpu_lowering, kernel_generator)
    else:
        raise ValueError(f'Unsupported Warp GPU lowering version: {version}')
    mlir.register_lowering(primitive, rule, platform="cuda")


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
