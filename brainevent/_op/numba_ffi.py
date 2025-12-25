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

import ctypes
import importlib.util
from typing import Dict, Sequence, Tuple

import jax
import numpy as np
from jax.interpreters import mlir

from brainevent._typing import KernelGenerator
from .util import OutType, abstract_arguments

__all__ = [
    'numba_cpu_ffi_rule',
]

numba_installed = importlib.util.find_spec('numba') is not None
if numba_installed:
    from numba import types, carray, cfunc

_NUMBA_CPU_FFI_HANDLES: Dict[str, object] = {}
_FFI_CALLBACK_COUNTER = 0


def _ensure_sequence(outs: OutType):
    if isinstance(outs, Sequence):
        return tuple(outs)
    return (outs,)


def _normalize_shapes_and_dtypes(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[object],
    kind: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[np.dtype, ...]]:
    if len(shapes) != len(dtypes):
        raise ValueError(f'Number of {kind} shapes ({len(shapes)}) must match number of dtypes ({len(dtypes)}).')
    normalized_shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
    normalized_dtypes = tuple(np.dtype(dtype) for dtype in dtypes)
    return normalized_shapes, normalized_dtypes


def _register_numba_cpu_ffi_target(
    kernel,
    input_shapes: Tuple[Tuple[int, ...], ...],
    input_dtypes: Tuple[np.dtype, ...],
    output_shapes: Tuple[Tuple[int, ...], ...],
    output_dtypes: Tuple[np.dtype, ...],
):
    global _FFI_CALLBACK_COUNTER

    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    # Create the callback function that processes the FFI call
    def ffi_callback(output_ptrs, input_ptrs):
        # Convert input pointers to numpy arrays using numba carray
        inputs = []
        for i in range(len(input_shapes)):
            ptr = ctypes.cast(input_ptrs[i], ctypes.c_void_p).value
            arr = carray(ptr, input_shapes[i], dtype=input_dtypes[i])
            inputs.append(arr)

        # Convert output pointers to numpy arrays
        outputs = []
        if len(output_shapes) > 1:
            for i in range(len(output_shapes)):
                ptr = ctypes.cast(output_ptrs[i], ctypes.c_void_p).value
                arr = carray(ptr, output_shapes[i], dtype=output_dtypes[i])
                outputs.append(arr)
        else:
            ptr = ctypes.cast(output_ptrs, ctypes.c_void_p).value
            arr = carray(ptr, output_shapes[0], dtype=output_dtypes[0])
            outputs.append(arr)

        # Call the kernel function
        kernel(*inputs, *outputs)

    # Define the ctypes signature for the FFI callback
    #
    # JAX FFI expects:
    #
    #   void(void* output_ptrs, void** input_ptrs) for single output
    #
    # or
    #
    #   void(void** output_ptrs, void** input_ptrs) for multiple outputs
    #
    if len(output_shapes) > 1:
        ffi_c_call_func = ctypes.CFUNCTYPE(
            None,  # return type: void
            ctypes.POINTER(ctypes.c_void_p),  # output_ptrs: void**
            ctypes.POINTER(ctypes.c_void_p),  # input_ptrs: void**
        )
    else:
        ffi_c_call_func = ctypes.CFUNCTYPE(
            None,  # return type: void
            ctypes.c_void_p,  # output_ptrs: void*
            ctypes.POINTER(ctypes.c_void_p),  # input_ptrs: void**
        )

    # Create the callback wrapper
    callback_func = ffi_c_call_func(ffi_callback)

    # Get the function pointer address
    ffi_ccall_address = ctypes.cast(callback_func, ctypes.c_void_p)

    # Create the PyCapsule and register the FFI target
    ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)

    target_name = f'brainevent_numba_ffi_{_FFI_CALLBACK_COUNTER}'
    _FFI_CALLBACK_COUNTER += 1

    jax.ffi.register_ffi_target(target_name, ffi_capsule, platform="cpu")

    # Keep the callback alive to prevent garbage collection
    _NUMBA_CPU_FFI_HANDLES[target_name] = callback_func

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )
    return target_name, out_types


def numba_cpu_ffi_rule(
    kernel_generator: KernelGenerator,
):
    from .op_numba import NumbaKernel

    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    def kernel_fn(*ins, outs: OutType, **kwargs):
        kernel = kernel_generator(**kwargs)
        assert isinstance(kernel, NumbaKernel), f'The kernel should be of type NumbaKernel, but got {type(kernel)}'
        input_output_aliases = kernel.input_output_aliases if kernel.input_output_aliases else {}

        # output information
        outs_seq = _ensure_sequence(outs)
        output_shapes, output_dtypes = _normalize_shapes_and_dtypes(
            tuple(out.shape for out in outs_seq),
            tuple(out.dtype for out in outs_seq),
            'output',
        )

        # input information
        in_info, _ = abstract_arguments(ins)
        input_shapes, input_dtypes = _normalize_shapes_and_dtypes(
            tuple(inp.shape for inp in in_info),
            tuple(inp.dtype for inp in in_info),
            'input',
        )

        # register FFI target
        target_name, out_types = _register_numba_cpu_ffi_target(
            kernel.kernel, input_shapes, input_dtypes, output_shapes, output_dtypes,
        )

        # call FFI with api_version=0 for old-style custom_call interface
        return jax.ffi.ffi_call(
            target_name, out_types, input_output_aliases=input_output_aliases, custom_call_api_version=0
        )(*ins)

    return mlir.lower_fun(kernel_fn, multiple_results=True)
