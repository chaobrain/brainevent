# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Runtime layer: CompiledModule, JAX FFI registration, dtype and attr utilities."""

import ctypes

import jax
import ml_dtypes
import numpy as np

from ._errors import BEError, RegistrationError


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

# Must match BE::DType enum in tensor.h
_JAX_TO_JKB: dict[np.dtype, int] = {
    np.dtype("float16"): 0,  # Float16
    np.dtype("float32"): 1,  # Float32
    np.dtype("float64"): 2,  # Float64
    np.dtype(ml_dtypes.bfloat16): 3,  # bfloat16 lives in ml_dtypes (JAX's dependency)
    np.dtype("int8"): 4,  # Int8
    np.dtype("int16"): 5,  # Int16
    np.dtype("int32"): 6,  # Int32
    np.dtype("int64"): 7,  # Int64
    np.dtype("uint8"): 8,  # UInt8
    np.dtype("uint16"): 9,  # UInt16
    np.dtype("uint32"): 10,  # UInt32
    np.dtype("uint64"): 11,  # UInt64
    np.dtype("bool"): 12,  # Bool
    np.dtype("complex64"): 13,  # Complex64
    np.dtype("complex128"): 14,  # Complex128
}

_BE_TO_JAX: dict[int, np.dtype] = {v: k for k, v in _JAX_TO_JKB.items()}

# Element byte widths (redundant with C++ side, but useful in Python)
DTYPE_SIZES: dict[int, int] = {
    0: 2, 1: 4, 2: 8, 3: 2,  # float types
    4: 1, 5: 2, 6: 4, 7: 8,  # signed int
    8: 1, 9: 2, 10: 4, 11: 8,  # unsigned int
    12: 1, 13: 8, 14: 16,  # bool, complex
}


def jax_dtype_to_jkb(dtype) -> int:
    """Convert a JAX/NumPy dtype to a BE DType enum value."""
    dtype = np.dtype(dtype)
    if dtype not in _JAX_TO_JKB:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return _JAX_TO_JKB[dtype]


def be_to_jax_dtype(jkb_dtype: int) -> np.dtype:
    """Convert a BE DType enum value to a NumPy dtype."""
    if jkb_dtype not in _BE_TO_JAX:
        raise ValueError(f"Unknown BE dtype enum value: {jkb_dtype}")
    return _BE_TO_JAX[jkb_dtype]


# ---------------------------------------------------------------------------
# Attribute type mapping
# ---------------------------------------------------------------------------

# Maps BE attr type name → numpy dtype used when passing from Python.
# For float16/bfloat16 use numpy.uint16 containing the raw 16-bit pattern.
ATTR_NUMPY_DTYPE: dict[str, type] = {
    "bool": bool,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float16": np.uint16,  # pass raw bits: np.float16(x).view(np.uint16)
    "bfloat16": np.uint16,  # pass raw bits: bfloat16_val.view(np.uint16)
    "float32": np.float32,
    "float64": np.float64,
    # complex64 / complex128 omitted: JAX mlir.ir_attribute cannot encode them.
}

# Legacy alias kept for backward compatibility.
ATTR_TYPES = {
    k: {"cpp": v, "python": ATTR_NUMPY_DTYPE[k]}
    for k, v in {
        "bool": "bool",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "int32": "int32_t",
        "uint32": "uint32_t",
        "int64": "int64_t",
        "uint64": "uint64_t",
        "float16": "uint16_t",
        "bfloat16": "uint16_t",
        "float32": "float",
        "float64": "double",
    }.items()
}


def validate_attr_value(name: str, value, expected_type: str) -> None:
    """Validate that a Python attribute value is compatible with *expected_type*."""
    info = ATTR_TYPES.get(expected_type)
    if info is None:
        raise TypeError(
            f"Unknown attribute type '{expected_type}' for '{name}'. "
            f"Supported: {list(ATTR_TYPES)}"
        )


# ---------------------------------------------------------------------------
# CompiledModule
# ---------------------------------------------------------------------------

class CompiledModule:
    """A compiled module loaded from a shared library.

    Each function listed at compilation time has a corresponding
    ``extern "C"`` symbol ``be_<name>`` in the ``.so``.  These are
    loaded via *ctypes* and wrapped for use with the JAX FFI system.

    Parameters
    ----------
    so_path : str
        Path to the compiled ``.so`` shared library.
    function_names : list[str]
        User function names whose FFI handler symbols to resolve.
    """

    def __init__(self, so_path: str, function_names: list[str]):
        self._so_path = str(so_path)
        self._lib = ctypes.CDLL(self._so_path)
        self._functions: dict[str, ctypes._CFuncPtr] = {}

        for fname in function_names:
            symbol = f"be_{fname}"
            try:
                fn = getattr(self._lib, symbol)
            except AttributeError:
                raise BEError(
                    f"Symbol '{symbol}' not found in {so_path}. "
                    f"Available symbols may not include the FFI wrapper for "
                    f"'{fname}'. Did the compilation succeed?"
                )
            # XLA FFI handler signature: XLA_FFI_Error*(XLA_FFI_CallFrame*)
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
            self._functions[fname] = fn

    def get_handler(self, name: str):
        """Return the ctypes function pointer for an FFI handler.

        Parameters
        ----------
        name : str
            User function name (without the ``be_`` prefix).
        """
        if name not in self._functions:
            raise KeyError(
                f"Function '{name}' not found in module. "
                f"Available: {list(self._functions)}"
            )
        return self._functions[name]

    @property
    def path(self) -> str:
        """Path to the loaded ``.so``."""
        return self._so_path

    @property
    def function_names(self) -> list[str]:
        """Names of available functions."""
        return list(self._functions)

    def __repr__(self) -> str:
        return f"CompiledModule(path={self._so_path!r}, functions={self.function_names})"


# ---------------------------------------------------------------------------
# JAX FFI registration bridge
# ---------------------------------------------------------------------------

# Global registry of (target_name → CompiledModule) to prevent garbage
# collection of the ctypes CDLL while the FFI target is alive.
_LIVE_MODULES: dict[str, CompiledModule] = {}

# Track registered names to give clear errors on duplicates.
_REGISTERED_TARGETS: set[str] = set()


def register_ffi_target(
    target_name: str,
    module: CompiledModule,
    func_name: str,
    *,
    platform: str = "CUDA",
) -> None:
    """Register a compiled function as a JAX FFI target.

    After registration, the function can be invoked inside ``@jax.jit``
    via ``jax.ffi.ffi_call(target_name, ...)``.

    Parameters
    ----------
    target_name : str
        Globally unique FFI target identifier.
    module : CompiledModule
        The loaded module containing the function.
    func_name : str
        Function name within the module.
    platform : str
        Target platform (``"CUDA"`` or ``"cpu"``).
    """
    if target_name in _REGISTERED_TARGETS:
        raise RegistrationError(f"FFI target '{target_name}' is already registered.")

    fn_ptr = module.get_handler(func_name)
    capsule = jax.ffi.pycapsule(fn_ptr)
    jax.ffi.register_ffi_target(target_name, capsule, platform=platform)

    # Keep the module alive
    _LIVE_MODULES[target_name] = module
    _REGISTERED_TARGETS.add(target_name)


def list_registered_targets() -> list[str]:
    """Return a sorted list of all registered FFI target names."""
    return sorted(_REGISTERED_TARGETS)
