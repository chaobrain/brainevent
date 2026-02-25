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

"""Source preprocessing and FFI wrapper code generation."""

import re
from dataclasses import dataclass, field

from ._errors import BEError

# C++ scalar type → BE attr type name.
# Only types that have a registered XLA_FFI_REGISTER_SCALAR_ATTR_DECODING
# (or uint16_t stand-in for float16/bfloat16) are listed.
_CPP_TYPE_TO_ATTR: dict[str, str] = {
    # bool
    "bool": "bool",
    # 8-bit
    "int8_t": "int8",
    "char": "int8",
    "uint8_t": "uint8",
    "unsigned char": "uint8",
    # 16-bit
    "int16_t": "int16",
    "short": "int16",
    "uint16_t": "uint16",
    "unsigned short": "uint16",
    # 32-bit
    "int32_t": "int32",
    "int": "int32",
    "uint32_t": "uint32",
    "unsigned int": "uint32",
    # 64-bit
    "int64_t": "int64",
    "long long": "int64",
    "uint64_t": "uint64",
    "unsigned long long": "uint64",
    # floating point
    "float": "float32",
    "double": "float64",
    # NOTE: std::complex<float> / std::complex<double> are NOT listed here
    # because JAX's Python FFI layer (mlir.ir_attribute) cannot encode complex
    # scalars.  Pass complex-valued attrs as two separate float re/im attrs.
    # NOTE: __half / __nv_bfloat16 are NOT listed here because they require a
    # uint16_t bit-cast that the simple call-through wrapper cannot generate
    # automatically.  Use the explicit form "attr.x:float16" / "attr.x:bfloat16"
    # instead; the impl function will receive uint16_t containing the raw bits.
}


def _infer_attr_type_from_source(
    source: str, func_name: str, attr_name: str
) -> str:
    """Return the BE attr type for *attr_name* by parsing the C++ signature.

    Raises BEError if the parameter is not found or its type is not a
    recognised scalar attr type.
    """
    pattern = rf'void\s+{re.escape(func_name)}\s*\(([^)]*)\)'
    m = re.search(pattern, source, re.DOTALL)
    if m is None:
        raise BEError(
            f"Cannot find 'void {func_name}(...)' in source to infer "
            f"type of attr '{attr_name}'. "
            f"Use the explicit form 'attr.{attr_name}:<type>' instead."
        )
    for raw_param in m.group(1).split(','):
        param = raw_param.strip()
        if not param:
            continue
        parts = param.split()
        if not parts:
            continue
        # Variable name is the last word; '*'/'&' may be attached to it
        # (C style: "float *ptr") or to the type (C++ style: "float* ptr").
        last_part = parts[-1]
        param_name = last_part.lstrip('*&')
        if param_name != attr_name:
            continue
        # Type is everything before the variable name.
        cpp_type = ' '.join(parts[:-1]).strip()
        # Reject pointer / reference types — they cannot be passed as XLA FFI
        # scalar attrs.  A '*' or '&' may appear in the type token ("float*")
        # or at the start of the variable token ("*ptr").
        is_ptr_or_ref = ('*' in cpp_type or '&' in cpp_type
                         or last_part != param_name)
        if is_ptr_or_ref:
            raise BEError(
                f"Cannot map C++ type '{cpp_type}' of '{attr_name}' in "
                f"'{func_name}' to a BE attr type. "
                f"Pointer and reference types cannot be passed as XLA FFI "
                f"scalar attributes. "
                f"Use the explicit form 'attr.{attr_name}:<type>' instead."
            )
        cpp_type = cpp_type.rstrip('*&')
        # Strip leading 'const' qualifier — scalars are passed by value.
        if cpp_type.startswith("const "):
            cpp_type = cpp_type[6:].strip()
        be_type = _CPP_TYPE_TO_ATTR.get(cpp_type)
        if be_type is None:
            raise BEError(
                f"Cannot map C++ type '{cpp_type}' of '{attr_name}' in "
                f"'{func_name}' to a BE attr type. "
                f"Supported C++ types: {list(_CPP_TYPE_TO_ATTR)}. "
                f"Use the explicit form 'attr.{attr_name}:<type>' instead."
            )
        return be_type
    raise BEError(
        f"Parameter '{attr_name}' not found in signature of '{func_name}'. "
        f"Use the explicit form 'attr.{attr_name}:<type>' instead."
    )


def resolve_bare_attr_types(
    tokens: list[str], func_name: str, source: str
) -> list[str]:
    """Replace bare ``"attr.<name>"`` tokens with ``"attr.<name>:<type>"``
    by parsing the C++ function signature in *source*.

    Fully-typed tokens (``"attr.alpha:float32"``) are passed through unchanged.
    """
    resolved: list[str] = []
    for token in tokens:
        if _ATTR_RE_BARE.match(token):
            attr_name = token[len("attr."):]
            attr_type = _infer_attr_type_from_source(source, func_name, attr_name)
            resolved.append(f"attr.{attr_name}:{attr_type}")
        else:
            resolved.append(token)
    return resolved


def infer_arg_spec_from_source(source: str, func_name: str) -> list[str]:
    """Infer an arg_spec by parsing the C++ function signature.

    Recognised parameter patterns:

    - ``const BE::Tensor param`` → ``"arg"`` (read-only input)
    - ``BE::Tensor param``       → ``"ret"`` (pre-allocated output)
    - ``int64_t stream``          → ``"stream"`` (CUDA stream handle)
    - ``float / double / int32_t / int64_t / int / bool param``
      → ``"attr.<name>:<type>"`` (scalar attribute, auto-typed)

    Raises
    ------
    BEError
        If the signature cannot be found or no output (non-const) is detected.
    """
    pattern = rf'void\s+{re.escape(func_name)}\s*\(([^)]*)\)'
    m = re.search(pattern, source, re.DOTALL)
    if m is None:
        raise BEError(
            f"Cannot find 'void {func_name}(...)' in source. "
            f"For explicit control use: functions={{'{func_name}': ['arg', 'ret', ...]}}"
        )

    params_str = m.group(1)
    tokens: list[str] = []
    for raw_param in params_str.split(','):
        param = raw_param.strip()
        if not param:
            continue
        if 'Tensor' in param:
            tv_idx = param.index('Tensor')
            is_const = 'const' in param[:tv_idx]
            tokens.append('arg' if is_const else 'ret')
        elif re.match(r'int64_t\s+stream\b', param):
            tokens.append('stream')
        else:
            # Try to detect a scalar attribute from known C++ types.
            parts = param.split()
            if len(parts) >= 2:
                last_part = parts[-1]
                param_name = last_part.lstrip('*&')
                cpp_type = ' '.join(parts[:-1]).strip()
                # Skip pointer / reference types — they cannot be XLA FFI
                # scalar attrs.  A '*'/'&' may appear in the type token
                # ("float*") or at the start of the variable token ("*ptr").
                is_ptr_or_ref = ('*' in cpp_type or '&' in cpp_type
                                 or last_part != param_name)
                if is_ptr_or_ref:
                    continue  # silently skip; not a scalar attr
                cpp_type = cpp_type.rstrip('*&')
                if cpp_type.startswith("const "):
                    cpp_type = cpp_type[6:].strip()
                be_type = _CPP_TYPE_TO_ATTR.get(cpp_type)
                if be_type is not None:
                    tokens.append(f'attr.{param_name}:{be_type}')
                # Unknown types are silently skipped.

    if not tokens:
        raise BEError(
            f"No Tensor parameters found in '{func_name}'. "
            "Cannot auto-detect arg_spec."
        )
    if 'ret' not in tokens:
        raise BEError(
            f"No non-const Tensor output found in '{func_name}'. "
            "Mark input Tensors with 'const' to distinguish inputs from "
            "outputs, or use the explicit dict form: "
            f"functions={{'{func_name}': ['arg', 'ret', ...]}}"
        )
    return tokens


# -- Annotation-based function discovery -------------------------------------

_BE_ANNOTATION_RE = re.compile(r'//\s*@BE\s+(\w+)')


def parse_be_annotations(source: str) -> dict[str, list[str]]:
    """Scan source for ``// @BE function_name`` annotations.

    Each annotation marks a user function for FFI export.  The arg_spec is
    auto-inferred from the C++ signature that follows the annotation.

    Example::

        // @BE vector_add
        void vector_add(BE::Tensor a, BE::Tensor b,
                         BE::Tensor out, int64_t stream) { ... }

    Returns
    -------
    dict[str, list[str]]
        Mapping from function name to inferred arg_spec tokens.

    Raises
    ------
    BEError
        If no annotations are found, or if inference fails for any annotated
        function.
    """
    names = _BE_ANNOTATION_RE.findall(source)
    if not names:
        raise BEError(
            "No '// @BE <function_name>' annotations found in source. "
            "Either add annotations or pass an explicit 'functions' dict."
        )

    seen: set[str] = set()
    result: dict[str, list[str]] = {}
    for name in names:
        if name in seen:
            raise BEError(f"Duplicate @BE annotation for '{name}'")
        seen.add(name)
        result[name] = infer_arg_spec_from_source(source, name)
    return result


# ---------------------------------------------------------------------------
# jax-tvm-ffi token aliases — normalize to canonical BE form
# ---------------------------------------------------------------------------

# Simple one-to-one aliases
_TOKEN_ALIASES: dict[str, str] = {
    "args": "arg",
    "rets": "ret",
    "ctx.stream": "stream",
}

# attrs.name → attr.name  (jax-tvm-ffi uses dot-separated 'attrs' prefix)
_ATTRS_TVM_RE = re.compile(r"^attrs\.(\w+)$")


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Normalise jax-tvm-ffi compatible tokens to canonical BE form.

    The following aliases are accepted and converted transparently:

    +--------------------+------------------+
    | jax-tvm-ffi token  | BE canonical    |
    +====================+==================+
    | ``"args"``         | ``"arg"``        |
    | ``"rets"``         | ``"ret"``        |
    | ``"ctx.stream"``   | ``"stream"``     |
    | ``"attrs.name"``   | ``"attr.name"``  |
    +--------------------+------------------+

    All other tokens (including fully-typed ``"attr.name:type"`` forms) are
    returned unchanged.

    Parameters
    ----------
    tokens : list[str]
        Raw arg_spec tokens, possibly using jax-tvm-ffi conventions.

    Returns
    -------
    list[str]
        Tokens in canonical BE form.
    """
    result: list[str] = []
    for token in tokens:
        if token in _TOKEN_ALIASES:
            result.append(_TOKEN_ALIASES[token])
        else:
            m = _ATTRS_TVM_RE.match(token)
            if m:
                result.append(f"attr.{m.group(1)}")
            else:
                result.append(token)
    return result


# Valid arg_spec tokens
_VALID_TOKENS = {"arg", "ret", "stream"}

# All supported attr type names (XLA FFI scalar decodings + uint16 stand-ins).
_ATTR_TYPES = (
    "bool",
    "int8", "uint8",
    "int16", "uint16",
    "int32", "uint32",
    "int64", "uint64",
    "float16", "bfloat16",  # raw uint16 bits — no native XLA FFI scalar
    "float32", "float64",
    # complex64 / complex128 are intentionally omitted: JAX's mlir.ir_attribute
    # cannot encode numpy complex scalars, so they are unusable as FFI attrs.
    # Use two separate float32/float64 re/im attrs instead.
)
_ATTR_RE = re.compile(
    r"^attr\.(\w+):(" + "|".join(_ATTR_TYPES) + r")$"
)
_ATTR_RE_BARE = re.compile(r"^attr\.(\w+)$")


@dataclass
class FunctionSpec:
    """Parsed specification for one user function."""
    name: str
    num_args: int = 0
    num_rets: int = 0
    has_stream: bool = False
    attrs: list[tuple[str, str]] = field(default_factory=list)
    # Order of *user* function params (what the generated wrapper must pass)
    user_param_order: list[str] = field(default_factory=list)


def parse_arg_spec(func_name: str, tokens: list[str]) -> FunctionSpec:
    """Parse an arg_spec token list into a FunctionSpec.

    Parameters
    ----------
    func_name : str
        Name of the user function.
    tokens : list[str]
        Per-parameter token list, e.g. ``["arg", "arg", "ret", "stream"]``.
        Each token maps to one parameter of the user C++ function:
        - ``"arg"``   → input ``BE::Tensor``
        - ``"ret"``   → output ``BE::Tensor``
        - ``"stream"`` → ``int64_t`` (CUDA stream handle)
        - ``"attr.<name>:<ctype>"`` → scalar attribute

    Returns
    -------
    FunctionSpec
    """
    spec = FunctionSpec(name=func_name)
    arg_idx = 0
    ret_idx = 0

    for token in tokens:
        if token == "arg":
            spec.num_args += 1
            spec.user_param_order.append(f"arg{arg_idx}")
            arg_idx += 1
        elif token == "ret":
            spec.num_rets += 1
            spec.user_param_order.append(f"ret{ret_idx}")
            ret_idx += 1
        elif token == "stream":
            if spec.has_stream:
                raise BEError(
                    f"Duplicate 'stream' token in arg_spec for {func_name}"
                )
            spec.has_stream = True
            spec.user_param_order.append("stream")
        else:
            m = _ATTR_RE.match(token)
            if m:
                attr_name, attr_type = m.group(1), m.group(2)
                spec.attrs.append((attr_name, attr_type))
                spec.user_param_order.append(f"attr_{attr_name}")
            else:
                raise BEError(
                    f"Invalid arg_spec token '{token}' for function "
                    f"'{func_name}'. Valid tokens: 'arg', 'ret', 'stream', "
                    f"'attr.<name>:<type>'"
                )

    if spec.num_rets == 0:
        raise BEError(
            f"arg_spec for '{func_name}' must contain at least one 'ret' token"
        )

    return spec


# -- Attribute C++ type mapping ------------------------------------------------
# _ATTR_CPP_TYPE : BE attr type → C++ type used in the generated impl function
# _ATTR_FFI_TYPE : BE attr type → C++ type used in the XLA FFI .Attr<T>() binding
#
# For float16 / bfloat16 there is no registered XLA_FFI_REGISTER_SCALAR_ATTR_DECODING
# for __half or __nv_bfloat16.  Both are stored as uint16_t raw bits.
# The user's C++ function must accept uint16_t and reinterpret internally, or
# the user must use float32/float64 if they want seamless scalar attr passing.

_ATTR_CPP_TYPE = {
    "bool": "bool",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float16": "uint16_t",  # raw f16 bits
    "bfloat16": "uint16_t",  # raw bf16 bits
    "float32": "float",
    "float64": "double",
}

_ATTR_FFI_TYPE = {
    "bool": "bool",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float16": "uint16_t",  # XLA FFI passes as U16 raw bits
    "bfloat16": "uint16_t",  # XLA FFI passes as U16 raw bits
    "float32": "float",
    "float64": "double",
}


def generate_ffi_wrapper(spec: FunctionSpec, allow_cuda_graph: bool = True) -> str:
    """Generate the XLA FFI wrapper C++ source for a single function.

    The generated code:
    1. Defines an ``extern "C"`` FFI handler symbol ``be_<name>``
    2. Inside the handler, converts ``ffi::AnyBuffer`` to ``BE::Tensor``
    3. Extracts the CUDA stream (if requested)
    4. Calls the user function with parameters in the order given by arg_spec

    Parameters
    ----------
    spec : FunctionSpec
        Parsed specification for the function.
    allow_cuda_graph : bool
        When ``True`` (default), embeds
        ``{xla::ffi::Traits::kCmdBufferCompatible}`` in the
        ``XLA_FFI_DEFINE_HANDLER_SYMBOL`` call so the kernel is unconditionally
        eligible for CUDA-graph / XLA command-buffer capture at the C++ level.
        Set to ``False`` for kernels with host-side side effects during replay.
    """
    name = spec.name
    impl_name = f"be_{name}_impl"
    handler_name = f"be_{name}"

    lines: list[str] = []
    lines.append(f"// ── FFI wrapper for {name} (auto-generated) ──")
    lines.append("")

    # ------------------------------------------------------------------
    # 1. Build the impl function signature.
    #    XLA FFI binding order: Ctx → Arg → Ret → Attr
    #    The impl function parameters must follow this exact order.
    # ------------------------------------------------------------------
    impl_params: list[str] = []

    # Ctx (stream) — always comes first if present
    if spec.has_stream:
        impl_params.append("cudaStream_t stream")

    # Args (input buffers)
    for i in range(spec.num_args):
        impl_params.append(f"xla::ffi::AnyBuffer arg{i}")

    # Rets (output buffers)
    for i in range(spec.num_rets):
        impl_params.append(f"xla::ffi::Result<xla::ffi::AnyBuffer> ret{i}")

    # Attrs (scalar attributes)
    for attr_name, attr_type in spec.attrs:
        cpp_type = _ATTR_CPP_TYPE[attr_type]
        impl_params.append(f"{cpp_type} attr_{attr_name}")

    sig = ",\n    ".join(impl_params)
    lines.append(f"xla::ffi::Error {impl_name}(")
    lines.append(f"    {sig}")
    lines.append(") {")

    # ------------------------------------------------------------------
    # 2. Convert buffers to Tensor
    # ------------------------------------------------------------------
    for i in range(spec.num_args):
        lines.append(
            f"    BE::Tensor tv_arg{i} = "
            f"BE::internal::buffer_to_tensor(arg{i});"
        )
    for i in range(spec.num_rets):
        lines.append(
            f"    BE::Tensor tv_ret{i} = "
            f"BE::internal::result_buffer_to_tensor(ret{i});"
        )

    # ------------------------------------------------------------------
    # 3. Call user function in the order specified by arg_spec
    # ------------------------------------------------------------------
    call_args: list[str] = []
    for p in spec.user_param_order:
        if p.startswith("arg"):
            call_args.append(f"tv_{p}")
        elif p.startswith("ret"):
            call_args.append(f"tv_{p}")
        elif p == "stream":
            call_args.append("reinterpret_cast<int64_t>(stream)")
        elif p.startswith("attr_"):
            call_args.append(p)  # same variable name

    lines.append(f"    {name}({', '.join(call_args)});")
    lines.append("    return xla::ffi::Error::Success();")
    lines.append("}")
    lines.append("")

    # ------------------------------------------------------------------
    # 4. XLA_FFI_DEFINE_HANDLER_SYMBOL — the extern "C" entry point
    # ------------------------------------------------------------------
    binding_parts: list[str] = []
    if spec.has_stream:
        binding_parts.append(
            "    .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()"
        )
    for _ in range(spec.num_args):
        binding_parts.append("    .Arg<xla::ffi::AnyBuffer>()")
    for _ in range(spec.num_rets):
        binding_parts.append("    .Ret<xla::ffi::AnyBuffer>()")
    for attr_name, attr_type in spec.attrs:
        ffi_type = _ATTR_FFI_TYPE[attr_type]
        binding_parts.append(f'    .Attr<{ffi_type}>("{attr_name}")')

    binding = "\n".join(binding_parts)
    lines.append(f"XLA_FFI_DEFINE_HANDLER_SYMBOL(")
    lines.append(f"    {handler_name},")
    lines.append(f"    {impl_name},")
    lines.append(f"    xla::ffi::Ffi::Bind()")
    if allow_cuda_graph:
        lines.append(f"{binding},")
        lines.append(f"    {{xla::ffi::Traits::kCmdBufferCompatible}}")
    else:
        lines.append(f"{binding}")
    lines.append(");")
    lines.append("")

    return "\n".join(lines)


def preprocess_source(
    user_source: str,
    function_specs: list[FunctionSpec],
    platform: str = "cuda",
    allow_cuda_graph: bool = True,
) -> str:
    """Build the final source file: preamble + user code + FFI wrappers.

    Parameters
    ----------
    user_source : str
        Verbatim user-written C++/CUDA source.
    function_specs : list[FunctionSpec]
        Parsed specifications for each function to wrap.
    platform : str
        ``"cuda"`` (default) or ``"cpu"``.  Controls which platform-specific
        headers are injected into the auto-generated preamble.
    allow_cuda_graph : bool
        Forwarded to :func:`generate_ffi_wrapper`.  When ``True``, embeds
        ``xla::ffi::Traits::kCmdBufferCompatible`` into every generated handler
        at the C++ level.  Default: ``True``.
    """
    parts: list[str] = []

    parts.append("// ════════════════════════════════════════════════════")
    parts.append("// Auto-generated by brainevent.source2kernel. Do not edit.")
    parts.append("// ════════════════════════════════════════════════════")
    parts.append("")
    # Internal header: XLA FFI ↔ Tensor bridge (platform-agnostic).
    # ffi_compat.h includes <cuda_runtime_api.h> under __CUDACC__ so the
    # generated wrapper's cudaStream_t reference compiles without the user
    # needing to add it manually.
    parts.append('#include "brainevent/ffi_compat.h"')
    parts.append("")

    # User source
    parts.append("// ── user source ─────────────────────────────────────")
    parts.append(user_source)
    parts.append("")

    # FFI wrappers
    parts.append("// ── FFI wrappers ───────────────────────────────────")
    for spec in function_specs:
        parts.append(generate_ffi_wrapper(spec, allow_cuda_graph=allow_cuda_graph))

    return "\n".join(parts)
