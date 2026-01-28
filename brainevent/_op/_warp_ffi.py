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
import enum
import importlib.util
import threading
import traceback
from typing import Sequence, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import mlir

from brainevent._typing import KernelGenerator
from ._util import OutType, abstract_arguments
from ._warp_util import get_dim, get_jax_device, generate_unique_name, check_warp_version

warp_installed = importlib.util.find_spec('warp') is not None

if warp_installed:
    try:
        import warp  # noqa: F401
        import warp._src.types
        import warp._src.context
    except Exception:
        warp_installed = False


# ============================================================================
# XLA FFI ctypes structures and enums
# (Based on XLA's C API: xla/ffi/api/c_api.h)
# ============================================================================


class XLA_FFI_Extension_Type(enum.IntEnum):
    Metadata = 1


class XLA_FFI_Extension_Base(ctypes.Structure):
    pass


XLA_FFI_Extension_Base._fields_ = [
    ("struct_size", ctypes.c_size_t),
    ("type", ctypes.c_int),
    ("next", ctypes.POINTER(XLA_FFI_Extension_Base)),
]


class XLA_FFI_DataType(enum.IntEnum):
    INVALID = 0
    PRED = 1
    S8 = 2
    S16 = 3
    S32 = 4
    S64 = 5
    U8 = 6
    U16 = 7
    U32 = 8
    U64 = 9
    F16 = 10
    F32 = 11
    F64 = 12
    BF16 = 16
    C64 = 15
    C128 = 18
    TOKEN = 17


class XLA_FFI_Buffer(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("dtype", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("rank", ctypes.c_int64),
        ("dims", ctypes.POINTER(ctypes.c_int64)),
    )


class XLA_FFI_Args(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("size", ctypes.c_int64),
        ("types", ctypes.POINTER(ctypes.c_int)),
        ("args", ctypes.POINTER(ctypes.c_void_p)),
    )


class XLA_FFI_Rets(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("size", ctypes.c_int64),
        ("types", ctypes.POINTER(ctypes.c_int)),
        ("rets", ctypes.POINTER(ctypes.c_void_p)),
    )


class XLA_FFI_ByteSpan(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.POINTER(ctypes.c_char)),
        ("len", ctypes.c_size_t),
    )


class XLA_FFI_Scalar(ctypes.Structure):
    _fields_ = (
        ("dtype", ctypes.c_int),
        ("value", ctypes.c_void_p),
    )


class XLA_FFI_Array(ctypes.Structure):
    _fields_ = (
        ("dtype", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    )


class XLA_FFI_AttrType(enum.IntEnum):
    ARRAY = 1
    DICTIONARY = 2
    SCALAR = 3
    STRING = 4


class XLA_FFI_Attrs(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("size", ctypes.c_int64),
        ("types", ctypes.POINTER(ctypes.c_int)),
        ("names", ctypes.POINTER(ctypes.POINTER(XLA_FFI_ByteSpan))),
        ("attrs", ctypes.POINTER(ctypes.c_void_p)),
    )


class XLA_FFI_Api_Version(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    )


class XLA_FFI_Handler_TraitsBits(enum.IntEnum):
    COMMAND_BUFFER_COMPATIBLE = 1 << 0


class XLA_FFI_Metadata(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("api_version", XLA_FFI_Api_Version),
        ("traits", ctypes.c_uint32),
    )


class XLA_FFI_Metadata_Extension(ctypes.Structure):
    _fields_ = (
        ("extension_base", XLA_FFI_Extension_Base),
        ("metadata", ctypes.POINTER(XLA_FFI_Metadata)),
    )


class XLA_FFI_Error_Code(enum.IntEnum):
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16


class XLA_FFI_Error_Create_Args(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("message", ctypes.c_char_p),
        ("errc", ctypes.c_int),
    )


XLA_FFI_Error_Create = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_Error_Create_Args))


class XLA_FFI_Stream_Get_Args(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("ctx", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
    )


XLA_FFI_Stream_Get = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_Stream_Get_Args))


class XLA_FFI_DeviceOrdinal_Get_Args(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("ctx", ctypes.c_void_p),
        ("device_ordinal", ctypes.c_int32),
    )


XLA_FFI_DeviceOrdinal_Get = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_DeviceOrdinal_Get_Args))


class XLA_FFI_Api(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("api_version", XLA_FFI_Api_Version),
        ("internal_api", ctypes.c_void_p),
        ("XLA_FFI_Error_Create", XLA_FFI_Error_Create),
        ("XLA_FFI_Error_GetMessage", ctypes.c_void_p),
        ("XLA_FFI_Error_Destroy", ctypes.c_void_p),
        ("XLA_FFI_Handler_Register", ctypes.c_void_p),
        ("XLA_FFI_Stream_Get", XLA_FFI_Stream_Get),
        ("XLA_FFI_TypeId_Register", ctypes.c_void_p),
        ("XLA_FFI_ExecutionContext_Get", ctypes.c_void_p),
        ("XLA_FFI_State_Set", ctypes.c_void_p),
        ("XLA_FFI_State_Get", ctypes.c_void_p),
        ("XLA_FFI_DeviceMemory_Allocate", ctypes.c_void_p),
        ("XLA_FFI_DeviceMemory_Free", ctypes.c_void_p),
        ("XLA_FFI_ThreadPool_Schedule", ctypes.c_void_p),
        ("XLA_FFI_ThreadPool_NumThreads", ctypes.c_void_p),
        ("XLA_FFI_Future_Create", ctypes.c_void_p),
        ("XLA_FFI_Future_SetAvailable", ctypes.c_void_p),
        ("XLA_FFI_Future_SetError", ctypes.c_void_p),
        ("XLA_FFI_RunId_Get", ctypes.c_void_p),
        ("XLA_FFI_DeviceOrdinal_Get", XLA_FFI_DeviceOrdinal_Get),
    )


class XLA_FFI_CallFrame(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
        ("api", ctypes.POINTER(XLA_FFI_Api)),
        ("ctx", ctypes.c_void_p),
        ("stage", ctypes.c_int),
        ("args", XLA_FFI_Args),
        ("rets", XLA_FFI_Rets),
        ("attrs", XLA_FFI_Attrs),
        ("future", ctypes.c_void_p),
    )


# XLA data type to jax dtype mapping
_xla_data_type_to_constructor = {
    XLA_FFI_DataType.PRED: jnp.bool,
    XLA_FFI_DataType.S8: jnp.int8,
    XLA_FFI_DataType.S16: jnp.int16,
    XLA_FFI_DataType.S32: jnp.int32,
    XLA_FFI_DataType.S64: jnp.int64,
    XLA_FFI_DataType.U8: jnp.uint8,
    XLA_FFI_DataType.U16: jnp.uint16,
    XLA_FFI_DataType.U32: jnp.uint32,
    XLA_FFI_DataType.U64: jnp.uint64,
    XLA_FFI_DataType.F16: jnp.float16,
    XLA_FFI_DataType.F32: jnp.float32,
    XLA_FFI_DataType.F64: jnp.float64,
    XLA_FFI_DataType.BF16: jnp.bfloat16,
    XLA_FFI_DataType.C64: jnp.complex64,
    XLA_FFI_DataType.C128: jnp.complex128,
}


# ============================================================================
# XLA FFI helper functions
# ============================================================================


def decode_bytespan(span: XLA_FFI_ByteSpan):
    length = span.len
    chars = ctypes.cast(span.ptr, ctypes.POINTER(ctypes.c_char * length))
    return chars.contents.value.decode("utf-8")


def decode_scalar(scalar: XLA_FFI_Scalar):
    dtype = jnp.dtype(_xla_data_type_to_constructor[scalar.dtype])
    raw_bytes = ctypes.string_at(scalar.value, dtype.itemsize)
    return np.frombuffer(raw_bytes, dtype=dtype).reshape(())


def decode_array(array: XLA_FFI_Array):
    dtype = jnp.dtype(_xla_data_type_to_constructor[array.dtype])
    raw_bytes = ctypes.string_at(array.data, dtype.itemsize * array.size)
    return np.frombuffer(raw_bytes, dtype=dtype)


def decode_attrs(attrs: XLA_FFI_Attrs):
    result = {}
    for i in range(attrs.size):
        attr_name = decode_bytespan(attrs.names[i].contents)
        attr_type = attrs.types[i]
        if attr_type == XLA_FFI_AttrType.STRING:
            bytespan = ctypes.cast(attrs.attrs[i], ctypes.POINTER(XLA_FFI_ByteSpan))
            attr_value = decode_bytespan(bytespan.contents)
        elif attr_type == XLA_FFI_AttrType.SCALAR:
            attr_value = ctypes.cast(attrs.attrs[i], ctypes.POINTER(XLA_FFI_Scalar))
            attr_value = decode_scalar(attr_value.contents)
        elif attr_type == XLA_FFI_AttrType.ARRAY:
            attr_value = ctypes.cast(attrs.attrs[i], ctypes.POINTER(XLA_FFI_Array))
            attr_value = decode_array(attr_value.contents)
        elif attr_type == XLA_FFI_AttrType.DICTIONARY:
            attr_value = ctypes.cast(attrs.attrs[i], ctypes.POINTER(XLA_FFI_Attrs))
            attr_value = decode_attrs(attr_value.contents)
        else:
            raise Exception("Unexpected attr type")
        result[attr_name] = attr_value
    return result


def create_ffi_error(api, errc, message):
    create_args = XLA_FFI_Error_Create_Args(
        ctypes.sizeof(XLA_FFI_Error_Create_Args),
        ctypes.POINTER(XLA_FFI_Extension_Base)(),
        ctypes.c_char_p(message.encode("utf-8")),
        errc,
    )
    return api.contents.XLA_FFI_Error_Create(create_args)


def get_stream_from_callframe(call_frame):
    api = call_frame.api
    get_stream_args = XLA_FFI_Stream_Get_Args(
        ctypes.sizeof(XLA_FFI_Stream_Get_Args),
        ctypes.POINTER(XLA_FFI_Extension_Base)(),
        call_frame.ctx,
        None,
    )
    api.contents.XLA_FFI_Stream_Get(get_stream_args)
    return get_stream_args.stream


def get_device_ordinal_from_callframe(call_frame):
    api = call_frame.api
    get_device_args = XLA_FFI_DeviceOrdinal_Get_Args(
        ctypes.sizeof(XLA_FFI_DeviceOrdinal_Get_Args),
        ctypes.POINTER(XLA_FFI_Extension_Base)(),
        call_frame.ctx,
        0,
    )
    api.contents.XLA_FFI_DeviceOrdinal_Get(get_device_args)
    return get_device_args.device_ordinal


def strides_from_shape(shape, dtype):
    """Compute C-contiguous strides from shape and warp dtype."""
    return warp._src.types.strides_from_shape(shape, dtype)


# ============================================================================
# FfiArg: describes a single FFI kernel argument
# ============================================================================


class FfiArg:
    def __init__(self, name, type, in_out=False):
        self.name = name
        self.type = type
        self.in_out = in_out
        self.is_array = isinstance(type, warp.array)

        if self.is_array:
            if hasattr(type.dtype, "_wp_scalar_type_"):
                self.dtype_shape = type.dtype._shape_
                self.dtype_ndim = len(self.dtype_shape)
                self.jax_scalar_type = warp.dtype_to_jax(type.dtype._wp_scalar_type_)
                self.jax_ndim = type.ndim + self.dtype_ndim
            elif type.dtype in warp._src.types.value_types:
                self.dtype_ndim = 0
                self.dtype_shape = ()
                self.jax_scalar_type = warp.dtype_to_jax(type.dtype)
                self.jax_ndim = type.ndim
            else:
                raise TypeError(
                    f"Invalid data type for array argument '{name}', "
                    f"expected scalar, vector, or matrix"
                )
            self.warp_ndim = type.ndim
        elif type in warp._src.types.value_types:
            self.dtype_ndim = 0
            self.dtype_shape = ()
            self.jax_scalar_type = warp.dtype_to_jax(warp._src.types.type_to_warp(type))
            self.jax_ndim = 0
            self.warp_ndim = 0
        else:
            raise TypeError(
                f"Invalid type for argument '{name}', "
                f"expected array or scalar, got {type}"
            )


# ============================================================================
# get_jax_output_type: compute JAX output ShapeDtypeStruct for an FfiArg
# ============================================================================


def get_jax_output_type(arg, dims):
    if isinstance(dims, int):
        dims = (dims,)

    ndim = len(dims)

    if arg.dtype_ndim > 0:
        # vector/matrix array
        if ndim == arg.warp_ndim:
            return jax.ShapeDtypeStruct((*dims, *arg.dtype_shape), arg.jax_scalar_type)
        elif ndim == arg.jax_ndim:
            # make sure inner dimensions match
            inner_dims = dims[-arg.dtype_ndim:]
            for i in range(arg.dtype_ndim):
                if inner_dims[i] != arg.dtype_shape[i]:
                    raise ValueError(f"Invalid output dimensions for argument '{arg.name}': {dims}")
            return jax.ShapeDtypeStruct(dims, arg.jax_scalar_type)
        else:
            raise ValueError(f"Invalid output dimensions for argument '{arg.name}': {dims}")
    else:
        # scalar array
        if ndim != arg.warp_ndim:
            raise ValueError(f"Invalid output dimensions for argument '{arg.name}': {dims}")
        return jax.ShapeDtypeStruct(dims, arg.jax_scalar_type)


# ============================================================================
# FFI launch descriptor and kernel
# ============================================================================


class FfiLaunchDesc:
    def __init__(self, static_inputs, launch_dims):
        self.static_inputs = static_inputs
        self.launch_dims = launch_dims


_FFI_CALLBACK_LOCK = threading.Lock()

# Module-level list to prevent JaxFFIKernel instances (and their ctypes
# callbacks) from being garbage-collected while XLA still holds references
# to the registered FFI targets.
_LIVE_FFI_KERNELS = []


class JaxFFIKernel:
    def __init__(
        self,
        kernel,
        block_dim: int,
        launch_dims: Sequence[int],
        input_output_aliases: Dict[int, int],
        vmap_method: str,
        module_preload_mode: str = 'CURRENT_DEVICE',
    ):
        assert module_preload_mode in [
            'CURRENT_DEVICE', 'ALL_DEVICES'
        ], f"Unknown module_preload_mode '{module_preload_mode}'"

        # kernel metadata
        self.kernel = kernel
        self.name = f"brainevent_warp_ffi_{generate_unique_name(kernel.func)}"
        self.block_dim = block_dim
        self.vmap_method = vmap_method
        self.launch_dims = launch_dims
        self.module_preload_mode = module_preload_mode

        self.num_kernel_args = len(kernel.adj.args)
        self.num_inputs = None
        self.launch_id = 0
        self.launch_descriptors = {}
        self.launch_input_output = {}

        # Build input output aliases.
        self.input_output_aliases = input_output_aliases if input_output_aliases else {}

        # Register the FFI callback.
        ffi_c_call_func = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
        self.callback_func = ffi_c_call_func(lambda call_frame: self.ffi_callback(call_frame))
        ffi_ccall_address = ctypes.cast(self.callback_func, ctypes.c_void_p)
        ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
        jax.ffi.register_ffi_target(self.name, ffi_capsule, platform="CUDA")

        # Prevent this instance from being garbage-collected while XLA
        # still holds a pointer to the registered callback.
        _LIVE_FFI_KERNELS.append(self)

    def __call__(self, *args, outs: OutType, **kwargs):
        launch_id = self.launch_id

        # Determine num_inputs on first call, validate on subsequent calls.
        if self.num_inputs is None:
            self.num_inputs = len(args)
        else:
            if len(args) != self.num_inputs:
                raise ValueError('Inconsistent number of input arguments, expected '
                                 f'{self.num_inputs}, got {len(args)}')
        num_outputs = self.num_kernel_args - self.num_inputs

        # Build input args list from kernel argument metadata.
        input_args = []
        for i in range(self.num_inputs):
            arg_name = self.kernel.adj.args[i].label
            arg = FfiArg(arg_name, self.kernel.adj.args[i].type, False)
            input_args.append(arg)

        # Build output args list from kernel argument metadata.
        output_args = []
        for i in range(self.num_inputs, self.num_kernel_args):
            arg_name = self.kernel.adj.args[i].label
            arg = FfiArg(arg_name, self.kernel.adj.args[i].type, False)
            if not arg.is_array:
                raise TypeError("All output arguments must be arrays")
            output_args.append(arg)
        self.launch_input_output[launch_id] = (input_args, output_args)

        # Validate inputs (dtype, ndim, inner dims for arrays; static check for scalars).
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

        # Launch dimensions.
        if isinstance(self.launch_dims, int):
            launch_dims = (self.launch_dims,)
        else:
            launch_dims = tuple(self.launch_dims)

        # Output types from outs specification.
        out_types = []
        if isinstance(outs, dict):  # assume a dictionary of shapes keyed on argument name
            outs = [outs.get(output_arg.name) for output_arg in output_args]
        outs, tree = abstract_arguments(outs)
        for out, arg in zip(outs, output_args):
            out_types.append(get_jax_output_type(arg, out.shape))
        if len(out_types) != num_outputs:
            raise ValueError('Inconsistent number of output arguments, expected '
                             f'{num_outputs}, got {len(out_types)}')

        # Build the FFI call.
        call = jax.ffi.ffi_call(
            self.name, out_types, vmap_method=self.vmap_method,
            input_output_aliases=self.input_output_aliases,
        )

        # Preload modules on the specified devices.
        if self.module_preload_mode == 'CURRENT_DEVICE':
            device = warp.device_from_jax(get_jax_device())
            self.kernel.module.load(device)
        elif self.module_preload_mode == 'ALL_DEVICES':
            for d in jax.local_devices():
                try:
                    dev = warp.device_from_jax(d)
                except Exception:
                    # ignore unsupported devices like TPUs
                    continue
                # we only support CUDA devices for now
                if dev.is_cuda:
                    self.kernel.module.load(dev)
        else:
            raise ValueError(f"Unknown preload mode '{self.module_preload_mode}'")

        # Save launch data to be retrieved by callback.
        self.launch_descriptors[launch_id] = FfiLaunchDesc(static_inputs, launch_dims)
        self.launch_id += 1

        return call(*args, launch_id=launch_id)

    def ffi_callback(self, call_frame):
        try:
            # On the first call, XLA runtime will query the API version and traits
            # metadata using the |extension| field. Respond to that query
            # if the metadata extension is present.
            extension = call_frame.contents.extension_start
            if extension:
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
                # Retrieve call info.
                attrs = decode_attrs(call_frame.contents.attrs)
                launch_id = int(attrs["launch_id"])
                launch_desc = self.launch_descriptors[launch_id]
                input_args, output_args = self.launch_input_output[launch_id]

                num_inputs = call_frame.contents.args.size
                inputs = ctypes.cast(
                    call_frame.contents.args.args,
                    ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer))
                )

                num_outputs = call_frame.contents.rets.size
                outputs = ctypes.cast(
                    call_frame.contents.rets.rets,
                    ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer))
                )

                assert num_inputs == self.num_inputs
                assert num_outputs == self.num_kernel_args - self.num_inputs

                launch_bounds = warp._src.types.launch_bounds_t(launch_desc.launch_dims)

                # First kernel param is the launch bounds.
                kernel_params = (ctypes.c_void_p * (1 + self.num_kernel_args))()
                kernel_params[0] = ctypes.addressof(launch_bounds)

                arg_refs = []

                # Input args.
                for i, input_arg in enumerate(input_args):
                    if input_arg.is_array:
                        buffer = inputs[i].contents
                        shape = buffer.dims[: input_arg.type.ndim]
                        strides = strides_from_shape(shape, input_arg.type.dtype)
                        arg = warp._src.types.array_t(buffer.data, 0, input_arg.type.ndim, shape, strides)
                        kernel_params[i + 1] = ctypes.addressof(arg)
                        arg_refs.append(arg)
                    else:
                        # scalar argument, get stashed value
                        value = launch_desc.static_inputs[input_arg.name]
                        arg = input_arg.type._type_(value)
                        kernel_params[i + 1] = ctypes.addressof(arg)
                        arg_refs.append(arg)

                # Output args.
                for i, output_arg in enumerate(output_args):
                    buffer = outputs[i].contents
                    shape = buffer.dims[: output_arg.type.ndim]
                    strides = strides_from_shape(shape, output_arg.type.dtype)
                    arg = warp._src.types.array_t(buffer.data, 0, output_arg.type.ndim, shape, strides)
                    kernel_params[num_inputs + i + 1] = ctypes.addressof(arg)
                    arg_refs.append(arg)

                # Get device and stream.
                device = warp.get_cuda_device(get_device_ordinal_from_callframe(call_frame.contents))
                stream = get_stream_from_callframe(call_frame.contents)

                # Get kernel hooks.
                hooks = self.kernel.module.get_kernel_hooks(self.kernel, device)
                assert hooks.forward, "Failed to find kernel entry point."

                # Launch the kernel.
                warp._src.context.runtime.core.wp_cuda_launch_kernel(
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


def _ffi_gpu_lowering(kernel_generator: KernelGenerator, ctx, *args, **kwargs):
    def kernel_fn(*args, **kwargs):
        check_warp_version()

        from .op_warp import WarpKernel
        wp_kernel = kernel_generator(**kwargs)
        assert isinstance(wp_kernel, WarpKernel), "Kernel generator did not return a WarpKernel"
        block_dim, warp_dims = get_dim(wp_kernel, **kwargs)

        # Handle callable input_output_aliases.
        input_output_aliases = wp_kernel.input_output_aliases
        if callable(input_output_aliases):
            input_output_aliases = input_output_aliases(**kwargs)

        return JaxFFIKernel(
            kernel=wp_kernel.kernel,
            block_dim=block_dim,
            launch_dims=warp_dims,
            input_output_aliases=input_output_aliases,
            vmap_method=wp_kernel.vmap_method,
            module_preload_mode=wp_kernel.module_preload_mode,
        )(*args, **kwargs)

    return mlir.lower_fun(kernel_fn, multiple_results=True)(ctx, *args, **kwargs)
