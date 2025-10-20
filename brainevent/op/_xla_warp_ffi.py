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

import jax
from jax.interpreters import mlir
from packaging import version

from brainevent._typing import KernelGenerator
from ._xla_warp_util import get_dim

warp_installed = importlib.util.find_spec('warp') is not None

if warp_installed:
    import warp  # noqa: F401

    if version.parse(warp.__version__) < version.parse("1.10.0"):
        from warp.jax_experimental.ffi import (
            generate_unique_name, FfiArg, XLA_FFI_CallFrame, XLA_FFI_Extension_Type,
            XLA_FFI_Handler_TraitsBits, XLA_FFI_Metadata_Extension, XLA_FFI_Buffer,
            FfiLaunchDesc, decode_attrs, XLA_FFI_Error_Code,
            create_ffi_error,
        )
    else:
        from warp._src.jax_experimental.ffi import (
            generate_unique_name, FfiArg, XLA_FFI_CallFrame, XLA_FFI_Extension_Type,
            XLA_FFI_Handler_TraitsBits, XLA_FFI_Metadata_Extension, XLA_FFI_Buffer,
            FfiLaunchDesc, decode_attrs, XLA_FFI_Error_Code,
            create_ffi_error,
        )

_FFI_CALLBACK_LOCK = threading.Lock()


class JaxFFIKernel:
    def __init__(
        self,
        kernel,
        num_outputs,
        vmap_method,
        launch_dims,
        output_dims,
        in_out_argnames,
        module_preload_mode: str = 'CURRENT_DEVICE',
    ):
        self.kernel = kernel
        self.name = f"brainevent_warp_kernel_{generate_unique_name(kernel.func)}"
        self.num_outputs = num_outputs
        self.vmap_method = vmap_method
        self.launch_dims = launch_dims
        self.output_dims = output_dims
        self.module_preload_mode = module_preload_mode
        self.first_array_arg = None
        self.launch_id = 0
        self.launch_descriptors = {}

        in_out_argnames_list = in_out_argnames or []
        in_out_argnames = set(in_out_argnames_list)
        if len(in_out_argnames_list) != len(in_out_argnames):
            raise AssertionError("in_out_argnames must not contain duplicate names")

        self.num_kernel_args = len(kernel.adj.args)
        self.num_in_out = len(in_out_argnames)
        self.num_inputs = self.num_kernel_args - num_outputs + self.num_in_out
        if self.num_outputs < 1:
            raise ValueError("At least one output is required")
        if self.num_outputs > self.num_kernel_args:
            raise ValueError("Number of outputs cannot be greater than the number of kernel arguments")
        if self.num_outputs < self.num_in_out:
            raise ValueError("Number of outputs cannot be smaller than the number of in_out_argnames")

        # process input args
        self.input_args = []
        for i in range(self.num_inputs):
            arg_name = kernel.adj.args[i].label
            arg = FfiArg(arg_name, kernel.adj.args[i].type, arg_name in in_out_argnames)
            if arg_name in in_out_argnames:
                in_out_argnames.remove(arg_name)
            if arg.is_array:
                # keep track of the first input array argument
                if self.first_array_arg is None:
                    self.first_array_arg = i
            self.input_args.append(arg)

        # process output args
        self.output_args = []
        for i in range(self.num_inputs, self.num_kernel_args):
            arg_name = kernel.adj.args[i].label
            if arg_name in in_out_argnames:
                raise AssertionError(
                    f"Expected an output-only argument for argument {arg_name}."
                    " in_out arguments should be placed before output-only arguments."
                )
            arg = FfiArg(arg_name, kernel.adj.args[i].type, False)
            if not arg.is_array:
                raise TypeError("All output arguments must be arrays")
            self.output_args.append(arg)

        if in_out_argnames:
            raise ValueError(f"in_out_argnames: '{in_out_argnames}' did not match any function argument names.")

        # Build input output aliases.
        out_id = 0
        input_output_aliases = {}
        for in_id, arg in enumerate(self.input_args):
            if not arg.in_out:
                continue
            input_output_aliases[in_id] = out_id
            out_id += 1
        self.input_output_aliases = input_output_aliases

        # register the callback
        FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
        self.callback_func = FFI_CCALLFUNC(lambda call_frame: self.ffi_callback(call_frame))
        ffi_ccall_address = ctypes.cast(self.callback_func, ctypes.c_void_p)
        ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
        jax.ffi.register_ffi_target(self.name, ffi_capsule, platform="CUDA")

    def __call__(self, *args, output_dims=None, launch_dims=None, vmap_method=None):
        num_inputs = len(args)
        if num_inputs != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but got {num_inputs}")

        # default argument fallback
        if launch_dims is None:
            launch_dims = self.launch_dims
        if output_dims is None:
            output_dims = self.output_dims
        if vmap_method is None:
            vmap_method = self.vmap_method

        # output types
        out_types = []

        # process inputs
        static_inputs = {}
        for i in range(num_inputs):
            input_arg = self.input_args[i]
            input_value = args[i]
            if input_arg.is_array:
                # check dtype
                if input_value.dtype != input_arg.jax_scalar_type:
                    raise TypeError(
                        f"Invalid data type for array argument '{input_arg.name}', expected {input_arg.jax_scalar_type}, got {input_value.dtype}"
                    )
                # check ndim
                if input_value.ndim != input_arg.jax_ndim:
                    raise TypeError(
                        f"Invalid dimensionality for array argument '{input_arg.name}', expected {input_arg.jax_ndim} dimensions, got {input_value.ndim}"
                    )
                # check inner dims
                for d in range(input_arg.dtype_ndim):
                    if input_value.shape[input_arg.type.ndim + d] != input_arg.dtype_shape[d]:
                        raise TypeError(
                            f"Invalid inner dimensions for array argument '{input_arg.name}', expected {input_arg.dtype_shape}, got {input_value.shape[-input_arg.dtype_ndim:]}"
                        )
            else:
                # make sure scalar is not a traced variable, should be static
                if isinstance(input_value, jax.core.Tracer):
                    raise ValueError(f"Argument '{input_arg.name}' must be a static value")
                # stash the value to be retrieved by callback
                static_inputs[input_arg.name] = input_arg.type(input_value)

            # append in-out arg to output types
            if input_arg.in_out:
                out_types.append(get_jax_output_type(input_arg, input_value.shape))

        # launch dimensions
        if isinstance(launch_dims, int):
            launch_dims = (launch_dims,)
        else:
            launch_dims = tuple(launch_dims)

        # output shapes
        if isinstance(output_dims, dict):
            # assume a dictionary of shapes keyed on argument name
            for output_arg in self.output_args:
                dims = output_dims.get(output_arg.name)
                if dims is None:
                    raise ValueError(f"Missing output dimensions for argument '{output_arg.name}'")
                out_types.append(get_jax_output_type(output_arg, dims))
        else:
            if output_dims is None:
                # use launch dimensions
                output_dims = launch_dims
            elif isinstance(output_dims, int):
                output_dims = (output_dims,)
            # assume same dimensions for all outputs
            for output_arg in self.output_args:
                out_types.append(get_jax_output_type(output_arg, output_dims))

        call = jax.ffi.ffi_call(
            self.name,
            out_types,
            vmap_method=vmap_method,
            input_output_aliases=self.input_output_aliases,
        )

        # preload on the specified devices
        if self.module_preload_mode == 'CURRENT_DEVICE':
            device = warp.device_from_jax(_get_jax_device())
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
        launch_id = self.launch_id
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

                num_inputs = call_frame.contents.args.size
                inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

                num_outputs = call_frame.contents.rets.size
                outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

                assert num_inputs == self.num_inputs
                assert num_outputs == self.num_outputs

                launch_bounds = warp.types.launch_bounds_t(launch_desc.launch_dims)

                # first kernel param is the launch bounds
                kernel_params = (ctypes.c_void_p * (1 + self.num_kernel_args))()
                kernel_params[0] = ctypes.addressof(launch_bounds)

                arg_refs = []

                # input and in-out args
                for i, input_arg in enumerate(self.input_args):
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
                for i, output_arg in enumerate(self.output_args):
                    buffer = outputs[i + self.num_in_out].contents
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
                assert hooks.forward, "Failed to find kernel entry point"

                # launch the kernel
                warp.context.runtime.core.wp_cuda_launch_kernel(
                    device.context,
                    hooks.forward,
                    launch_bounds.size,
                    0,
                    256,
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

        return JaxFFIKernel(wp_kernel.kernel)(*args)

    return mlir.lower_fun(kernel_fn, multiple_results=True)
