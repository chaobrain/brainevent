# -*- coding: utf-8 -*-
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

"""Content-derived naming for XLA FFI targets.

Counter-based FFI target names (``brainevent_numba_ffi_{N}``) are
process-order-dependent: they break ``jax.export`` across processes and
silently rebind after a module reload (audit finding 14).  This module
derives a stable fingerprint from the *content* of a kernel — its bytecode,
constants, and closure values — so that the same kernel always registers
under the same target name, in any process, in any order.

Fingerprinting is best-effort: when a kernel closes over a value that cannot
be serialized deterministically, :func:`kernel_content_fingerprint` returns
``None`` and the caller must fall back to a per-process unique name (losing
cross-process stability for that kernel only, never correctness).
"""

import hashlib
import types
from typing import Any, Optional

import numpy as np

__all__ = [
    'kernel_content_fingerprint',
]

# Cap on the number of bytes of ndarray closure content folded into the
# fingerprint.  Larger arrays are refused (return None) rather than silently
# truncated, since a truncated hash could collide two genuinely different
# kernels.
_MAX_ARRAY_BYTES = 1 << 20  # 1 MiB


def _serialize_value(value: Any, parts: list, depth: int = 0) -> bool:
    """Append a deterministic byte representation of *value* to *parts*.

    Returns ``True`` on success, ``False`` if *value* has no deterministic
    representation (caller must then abandon fingerprinting).
    """
    if depth > 8:
        return False
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        parts.append(f'{type(value).__name__}:{value!r}'.encode())
        return True
    if isinstance(value, types.ModuleType):
        # Kernels frequently close over modules (e.g. ``from numba import
        # cuda`` in the defining scope); the module's qualified name is its
        # identity for fingerprinting purposes.
        parts.append(f'mod:{value.__name__}'.encode())
        return True
    if isinstance(value, types.CodeType):
        return _serialize_code(value, parts, depth + 1)
    if isinstance(value, np.dtype):
        parts.append(f'dtype:{value.str}'.encode())
        return True
    if isinstance(value, np.ndarray):
        if value.nbytes > _MAX_ARRAY_BYTES:
            return False
        parts.append(f'ndarray:{value.dtype.str}:{value.shape}'.encode())
        parts.append(np.ascontiguousarray(value).tobytes())
        return True
    if isinstance(value, np.generic):
        parts.append(f'npscalar:{value.dtype.str}:{value.item()!r}'.encode())
        return True
    if isinstance(value, (tuple, list, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, frozenset) else value
        parts.append(f'{type(value).__name__}[{len(items)}]:'.encode())
        return all(_serialize_value(v, parts, depth + 1) for v in items)
    if isinstance(value, dict):
        try:
            items = sorted(value.items(), key=lambda kv: repr(kv[0]))
        except Exception:
            return False
        parts.append(f'dict[{len(items)}]:'.encode())
        return all(
            _serialize_value(k, parts, depth + 1) and _serialize_value(v, parts, depth + 1)
            for k, v in items
        )
    if isinstance(value, types.FunctionType):
        return _serialize_function(value, parts, depth + 1)
    py_func = getattr(value, 'py_func', None)
    if isinstance(py_func, types.FunctionType):
        # Numba dispatchers (e.g. ``@cuda.jit`` device helpers captured in a
        # closure): fingerprint the wrapped Python function so editing the
        # helper's body changes the fingerprint.
        parts.append(b'dispatch:')
        return _serialize_function(py_func, parts, depth + 1)
    return False


def _serialize_code(code: types.CodeType, parts: list, depth: int = 0) -> bool:
    """Append a deterministic representation of a code object to *parts*."""
    parts.append(code.co_name.encode())
    parts.append(code.co_code)
    parts.append(repr(code.co_names).encode())
    parts.append(repr(code.co_varnames).encode())
    parts.append(repr(code.co_freevars).encode())
    parts.append(f'argcount:{code.co_argcount}:{code.co_kwonlyargcount}'.encode())
    return all(_serialize_value(const, parts, depth + 1) for const in code.co_consts)


def _collect_names(code: types.CodeType, acc: set) -> None:
    """Recursively collect ``co_names`` from *code* and nested code objects."""
    acc.update(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _collect_names(const, acc)


def _serialize_global(name: str, value: Any, parts: list, depth: int) -> bool:
    """Append a deterministic representation of a referenced global to *parts*.

    Kernels capture globals by *name* in bytecode, but numba freezes their
    *values* at compile time — so a fingerprint that ignored global values
    would give two behaviorally different kernels the same name.  Modules and
    named callables are identified by their qualified name; data values are
    serialized in full; anything without a deterministic representation
    aborts fingerprinting.
    """
    if isinstance(value, types.ModuleType):
        parts.append(f'gmod:{name}:{value.__name__}'.encode())
        return True
    if isinstance(value, types.FunctionType):
        parts.append(f'gfn:{name}:'.encode())
        return _serialize_function(value, parts, depth + 1)
    py_func = getattr(value, 'py_func', None)
    if isinstance(py_func, types.FunctionType):
        # Numba dispatchers (e.g. ``@cuda.jit`` device helpers called from the
        # kernel body): fingerprint the wrapped Python function, NOT just the
        # qualified name — otherwise editing the helper's body while keeping
        # its name would silently reuse the stale registered handler.
        parts.append(f'gdispatch:{name}:'.encode())
        return _serialize_function(py_func, parts, depth + 1)
    if callable(value):
        # Builtins and ufuncs: identify by qualified name (their behavior is
        # fixed by the interpreter/library version, not by user edits).
        mod = getattr(value, '__module__', '') or ''
        qual = getattr(value, '__qualname__', None) or getattr(value, '__name__', None)
        if qual is None:
            return False
        parts.append(f'gcall:{name}:{mod}:{qual}'.encode())
        return True
    parts.append(f'gval:{name}:'.encode())
    return _serialize_value(value, parts, depth + 1)


def _serialize_function(fn: types.FunctionType, parts: list, depth: int = 0) -> bool:
    """Append a deterministic representation of a Python function to *parts*."""
    if depth > 8:
        return False
    parts.append(f'{fn.__module__}:{fn.__qualname__}'.encode())
    if not _serialize_code(fn.__code__, parts, depth):
        return False
    if fn.__defaults__:
        parts.append(b'defaults:')
        if not all(_serialize_value(d, parts, depth + 1) for d in fn.__defaults__):
            return False
    if fn.__closure__:
        parts.append(b'closure:')
        for cell in fn.__closure__:
            try:
                content = cell.cell_contents
            except ValueError:  # empty cell
                parts.append(b'<empty-cell>')
                continue
            if not _serialize_value(content, parts, depth + 1):
                return False
    # Globals referenced by the kernel body (numba freezes their values at
    # compile time, so they are part of the kernel's content).
    names: set = set()
    _collect_names(fn.__code__, names)
    fn_globals = fn.__globals__
    for name in sorted(names):
        if name not in fn_globals:
            # Attribute names and true builtins: covered by co_code/co_names.
            continue
        if not _serialize_global(name, fn_globals[name], parts, depth):
            return False
    return True


def kernel_content_fingerprint(kernel: Any, extra: tuple = ()) -> Optional[str]:
    """Compute a stable, content-derived fingerprint for a kernel function.

    The fingerprint is a function of the kernel's qualified name, bytecode,
    constants (recursively, including nested code objects), argument layout,
    default values, closure cell values, and any *extra* caller-supplied
    discriminators (e.g. launch-mode signature, shared memory size).  Two
    textually identical kernels — including across processes or module
    reloads — produce the same fingerprint; kernels differing in code or
    captured values produce different ones.

    Parameters
    ----------
    kernel : callable
        The kernel to fingerprint.  Numba dispatchers (CPU ``@njit`` and
        ``numba.cuda`` kernels) are unwrapped to their underlying Python
        function via the ``py_func`` attribute when present.
    extra : tuple, optional
        Additional hashable discriminators folded into the fingerprint
        (serialized with the same deterministic scheme).

    Returns
    -------
    str or None
        A 16-character hexadecimal digest, or ``None`` when the kernel (or a
        closure value it captures) has no deterministic byte representation.
        Callers must treat ``None`` as "fall back to a per-process unique
        name"; they must not substitute a weaker hash.

    See Also
    --------
    brainevent._op.numba_ffi : CPU FFI target registration (consumer).
    brainevent._op.numba_cuda_ffi : CUDA FFI target registration (consumer).

    Examples
    --------
    .. code-block:: python

        >>> def add_one(x, out):
        ...     out[0] = x[0] + 1.0
        >>> fp = kernel_content_fingerprint(add_one)
        >>> fp is not None and len(fp) == 16
        True
    """
    fn = getattr(kernel, 'py_func', kernel)
    if not isinstance(fn, types.FunctionType):
        return None
    parts: list = []
    if not _serialize_function(fn, parts):
        return None
    if extra:
        parts.append(b'extra:')
        if not _serialize_value(tuple(extra), parts):
            return None
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b'\x00')
    return h.hexdigest()[:16]
