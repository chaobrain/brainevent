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
import functools
import importlib.util

from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from packaging import version

from brainevent._compatible_import import register_custom_call, custom_call
from brainevent._typing import KernelGenerator
from .warp_util import get_jax_device, get_dim

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
    device = warp.get_cuda_device(get_jax_device().id)

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

warp_cpu_CCALL_FUNC_single_out = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
)

warp_cpu_CCALL_FUNC_multi_outs = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
)


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
    register_custom_call("brainevent_warp_custom_call", warp_capsule, "gpu")


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


def _custom_call_gpu_lowering(
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
    wp_kernel = kernel_generator(**kwargs)
    assert isinstance(wp_kernel.kernel, warp.context.Kernel), (
        f'The kernel should be a Warp '
        f'kernel. But we got {wp_kernel}'
    )

    kernel_id = _register_warp_kernel(wp_kernel.kernel)
    block_dim, warp_dims = get_dim(wp_kernel, **kwargs)

    # TODO: This may not be necessary, but it is perhaps better not to be
    #       mucking with kernel loading while already running the workload.
    module = wp_kernel.kernel.module
    device = warp.device_from_jax(get_jax_device())
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
