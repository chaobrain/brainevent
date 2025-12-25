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

import re
from typing import Union, Callable

import jax

__all__ = []


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
