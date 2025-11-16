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

import importlib.util
import re
from typing import Union, Callable

import jax
import numpy as np

warp_installed = importlib.util.find_spec('warp') is not None

if warp_installed:
    import warp  # pylint: disable=import-error, import-outside-toplevel

__all__ = [
    'jaxtype_to_warptype',
    'jaxinfo_to_warpinfo',
]


# generates a C function name based on the python function name
def make_full_qualified_name(func: Union[str, Callable]) -> str:
    if not isinstance(func, str):
        func = func.__qualname__
    return re.sub("[^0-9a-zA-Z_]+", "", func.replace(".", "__"))


# ensure unique FFI callback names
ffi_name_counts = {}


def generate_unique_name(func) -> str:
    key = make_full_qualified_name(func)
    unique_id = ffi_name_counts.get(key, 0)
    ffi_name_counts[key] = unique_id + 1
    return f"{key}_{unique_id}"


def get_jax_device():
    # check if jax.default_device() context manager is active
    device = jax.config.jax_default_device
    # if default device is not set, use first device
    if device is None:
        device = jax.local_devices()[0]
    return device


def get_dim(wp_kernel, **kwargs):
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
        raise ValueError(f"Invalid block dimensions, expected int, got {block_dim}")

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

    return block_dim, warp_dims


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
