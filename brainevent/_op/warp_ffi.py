# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ctypes
import importlib.util
import threading
import traceback
from typing import Sequence, Dict

import jax
from jax.interpreters import mlir
from packaging import version

from brainevent._typing import KernelGenerator
from .util import OutType, abstract_arguments
from .warp_util import get_dim, get_jax_device, generate_unique_name

warp_installed = importlib.util.find_spec('warp') is not None

if warp_installed:
    import warp  # noqa: F401

    if version.parse(warp.__version__) < version.parse("1.10.0"):
        from warp.jax_experimental.ffi import (
            FfiArg, XLA_FFI_CallFrame, XLA_FFI_Extension_Type, XLA_FFI_Array, XLA_FFI_Error_Code,
            XLA_FFI_Handler_TraitsBits, XLA_FFI_Metadata_Extension, XLA_FFI_Buffer,
            decode_attrs, create_ffi_error, strides_from_shape, get_device_ordinal_from_callframe,
            get_stream_from_callframe, get_jax_output_type
        )

    else:
        from warp._src.jax_experimental.ffi import (
            FfiArg, XLA_FFI_CallFrame, XLA_FFI_Extension_Type, XLA_FFI_Error_Code,
            XLA_FFI_Handler_TraitsBits, XLA_FFI_Metadata_Extension, XLA_FFI_Buffer,
            decode_attrs, create_ffi_error, strides_from_shape, get_device_ordinal_from_callframe,
            get_stream_from_callframe, get_jax_output_type
        )


class FfiLaunchDesc:
    def __init__(self, static_inputs, launch_dims):
        self.static_inputs = static_inputs
        self.launch_dims = launch_dims


_FFI_CALLBACK_LOCK = threading.Lock()


class JaxFFIKernel:
    def __init__(
        self,
        kernel,
        vmap_method: str,
        block_dim: int,
        launch_dims: Sequence[int],
        input_output_aliases: Dict[int, int],
        module_preload_mode: str = 'CURRENT_DEVICE',
    ):
        assert module_preload_mode in [
            'CURRENT_DEVICE', 'ALL_DEVICES'
        ], f"Unknown module_preload_mode '{module_preload_mode}'"

        # parameters
        self.kernel = kernel
        self.num_kernel_args = len(kernel.adj.args)
        self.name = f"brainevent_warp_ffi_{generate_unique_name(kernel.func)}"
        self.block_dim = block_dim
        self.vmap_method = vmap_method
        self.launch_dims = launch_dims
        self.module_preload_mode = module_preload_mode
        self.launch_id = 0
        self.launch_descriptors = {}
        self.launch_input_output = {}
        self.num_inputs = None

        # Build input output aliases.
        self.input_output_aliases = input_output_aliases if input_output_aliases else {}

        # register the callback
        ffi_c_call_func = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
        self.callback_func = ffi_c_call_func(lambda call_frame: self.ffi_callback(call_frame))
        ffi_ccall_address = ctypes.cast(self.callback_func, ctypes.c_void_p)
        ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
        jax.ffi.register_ffi_target(self.name, ffi_capsule, platform="CUDA")

    def __call__(self, *args, outs: OutType):
        launch_id = self.launch_id
        if self.num_inputs is None:
            self.num_inputs = len(args)
        else:
            if len(args) != self.num_inputs:
                raise ValueError('Inconsistent number of input arguments, expected '
                                 f'{self.num_inputs}, got {len(args)}')
        num_outputs = self.num_kernel_args - self.num_inputs

        # process input args
        input_args = []
        for i in range(self.num_inputs):
            arg_name = self.kernel.adj.args[i].label
            arg = FfiArg(arg_name, self.kernel.adj.args[i].type, False)
            input_args.append(arg)

        # process output args
        output_args = []
        for i in range(self.num_inputs, self.num_kernel_args):
            arg_name = self.kernel.adj.args[i].label
            arg = FfiArg(arg_name, self.kernel.adj.args[i].type, False)
            if not arg.is_array:
                raise TypeError("All output arguments must be arrays")
            output_args.append(arg)
        self.launch_input_output[launch_id] = (input_args, output_args)

        # process inputs
        static_inputs = {}
        for i in range(self.num_inputs):
            input_arg = input_args[i]
            input_value = args[i]
            if input_arg.is_array:
                # check dtype
                if input_value.dtype != input_arg.jax_scalar_type:
                    raise TypeError(
                        f"Invalid data type for array argument '{input_arg.name}', "
                        f"expected {input_arg.jax_scalar_type}, "
                        f"got {input_value.dtype}"
                    )
                # check ndim
                if input_value.ndim != input_arg.jax_ndim:
                    raise TypeError(
                        f"Invalid dimensionality for array argument "
                        f"'{input_arg.name}', expected {input_arg.jax_ndim} "
                        f"dimensions, got {input_value.ndim}"
                    )
                # check inner dims
                for d in range(input_arg.dtype_ndim):
                    if input_value.shape[input_arg.type.ndim + d] != input_arg.dtype_shape[d]:
                        raise TypeError(
                            f"Invalid inner dimensions for array argument "
                            f"'{input_arg.name}', expected {input_arg.dtype_shape}, "
                            f"got {input_value.shape[-input_arg.dtype_ndim:]}"
                        )
            else:
                # make sure scalar is not a traced variable, should be static
                if isinstance(input_value, jax.core.Tracer):
                    raise ValueError(f"Argument '{input_arg.name}' must be a static value")
                # stash the value to be retrieved by callback
                static_inputs[input_arg.name] = input_arg.type(input_value)

        # launch dimensions
        if isinstance(self.launch_dims, int):
            launch_dims = (self.launch_dims,)
        else:
            launch_dims = tuple(self.launch_dims)

        # output types
        out_types = []
        if isinstance(outs, dict):  # assume a dictionary of shapes keyed on argument name
            outs = [outs.get(output_arg.name) for output_arg in output_args]
        outs, tree = abstract_arguments(outs)
        for out, arg in zip(outs, output_args):
            out_types.append(get_jax_output_type(arg, out.shape))
        if len(out_types) != num_outputs:
            raise ValueError('Inconsistent number of output arguments, expected '
                             f'{num_outputs}, got {len(out_types)}')

        # call FFI
        call = jax.ffi.ffi_call(
            self.name, out_types, vmap_method=self.vmap_method,
            input_output_aliases=self.input_output_aliases,
        )

        # preload on the specified devices
        if self.module_preload_mode == 'CURRENT_DEVICE':
            device = warp.device_from_jax(get_jax_device())
            self.kernel.module.load(device)
        elif self.module_preload_mode == 'ALL_DEVICES':
            for d in jax.local_devices():
                try:
                    dev = warp.device_from_jax(d)
                except Exception:
                    # ignore unsupported devices like TPUs
                    pass
                # we only support CUDA devices for now
                if dev.is_cuda:
                    self.kernel.module.load(dev)
        else:
            raise ValueError(f"Unknown preload mode '{self.module_preload_mode}'")

        # save launch data to be retrieved by callback
        self.launch_descriptors[launch_id] = FfiLaunchDesc(static_inputs, launch_dims)
        self.launch_id += 1

        return call(*args, launch_id=launch_id)

    def ffi_callback(self, call_frame):
        try:
            # On the first call, XLA runtime will query the API version and traits
            # metadata using the |extension| field. Let us respond to that query
            # if the metadata extension is present.
            extension = call_frame.contents.extension_start
            if extension:
                # Try to set the version metadata.
                if extension.contents.type == XLA_FFI_Extension_Type.Metadata:
                    metadata_ext = ctypes.cast(extension, ctypes.POINTER(XLA_FFI_Metadata_Extension))
                    metadata_ext.contents.metadata.contents.api_version.major_version = 0
                    metadata_ext.contents.metadata.contents.api_version.minor_version = 1
                    # Turn on CUDA graphs for this handler.
                    metadata_ext.contents.metadata.contents.traits = (
                        XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
                    )
                    return None

            # Lock is required to prevent race conditions when callback is invoked
            # from multiple threads, like with pmap.
            with _FFI_CALLBACK_LOCK:
                # retrieve call info
                attrs = decode_attrs(call_frame.contents.attrs)
                launch_id = int(attrs["launch_id"])
                launch_desc = self.launch_descriptors[launch_id]
                input_args, output_args = self.launch_input_output[launch_id]

                num_inputs = call_frame.contents.args.size
                inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

                num_outputs = call_frame.contents.rets.size
                outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

                assert num_inputs == self.num_inputs
                assert num_outputs == self.num_kernel_args - self.num_inputs

                launch_bounds = warp.types.launch_bounds_t(launch_desc.launch_dims)

                # first kernel param is the launch bounds
                kernel_params = (ctypes.c_void_p * (1 + self.num_kernel_args))()
                kernel_params[0] = ctypes.addressof(launch_bounds)

                arg_refs = []

                # input and in-out args
                for i, input_arg in enumerate(input_args):
                    if input_arg.is_array:
                        buffer = inputs[i].contents
                        shape = buffer.dims[: input_arg.type.ndim]
                        strides = strides_from_shape(shape, input_arg.type.dtype)
                        arg = warp.types.array_t(buffer.data, 0, input_arg.type.ndim, shape, strides)
                        kernel_params[i + 1] = ctypes.addressof(arg)
                        arg_refs.append(arg)  # keep a reference
                    else:
                        # scalar argument, get stashed value
                        value = launch_desc.static_inputs[input_arg.name]
                        arg = input_arg.type._type_(value)
                        kernel_params[i + 1] = ctypes.addressof(arg)
                        arg_refs.append(arg)  # keep a reference

                # pure output args (skip in-out FFI buffers)
                for i, output_arg in enumerate(output_args):
                    buffer = outputs[i].contents
                    shape = buffer.dims[: output_arg.type.ndim]
                    strides = strides_from_shape(shape, output_arg.type.dtype)
                    arg = warp.types.array_t(buffer.data, 0, output_arg.type.ndim, shape, strides)
                    kernel_params[num_inputs + i + 1] = ctypes.addressof(arg)
                    arg_refs.append(arg)  # keep a reference

                # get device and stream
                device = warp.get_cuda_device(get_device_ordinal_from_callframe(call_frame.contents))
                stream = get_stream_from_callframe(call_frame.contents)

                # get kernel hooks
                hooks = self.kernel.module.get_kernel_hooks(self.kernel, device)
                assert hooks.forward, "Failed to find kernel entry point. "

                # launch the kernel
                if version.parse(warp.__version__) >= version.parse("1.9.0"):
                    warp_launch_kernel_func = warp.context.runtime.core.wp_cuda_launch_kernel
                else:
                    warp_launch_kernel_func = warp.context.runtime.core.cuda_launch_kernel
                warp_launch_kernel_func(
                    device.context,
                    hooks.forward,
                    launch_bounds.size,
                    0,
                    self.block_dim,
                    hooks.forward_smem_bytes,
                    kernel_params,
                    stream,
                )

        except Exception as e:
            print(traceback.format_exc())
            return create_ffi_error(
                call_frame.contents.api,
                XLA_FFI_Error_Code.UNKNOWN,
                f"FFI callback error: {type(e).__name__}: {e}"
            )


def _ffi_gpu_lowering(
    kernel_generator: KernelGenerator,
):
    def kernel_fn(*args, **kwargs):
        wp_kernel = kernel_generator(**kwargs)  # ensure kernel is registered
        block_dim, warp_dims = get_dim(wp_kernel, **kwargs)

        return JaxFFIKernel(
            kernel=wp_kernel.kernel,
            vmap_method=kernel_generator.vmap_method,
            block_dim=block_dim,
            launch_dims=warp_dims,
            input_output_aliases=kernel_generator.input_output_aliases,
            module_preload_mode=kernel_generator.module_preload_mode,
        )(*args, **kwargs)

    return mlir.lower_fun(kernel_fn, multiple_results=True)
