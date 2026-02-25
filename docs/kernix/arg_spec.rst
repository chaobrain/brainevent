arg_spec System
===============

Every function compiled by ``brainevent.kernix`` needs an **arg_spec** — a
list of tokens that describes the function's parameter types.  This tells
kernix how to generate the XLA FFI wrapper code.

Token Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Token
     - Description
     - C++ Parameter Type
   * - ``"arg"``
     - Input tensor (read-only)
     - ``const BE::Tensor``
   * - ``"ret"``
     - Output tensor (pre-allocated by XLA)
     - ``BE::Tensor``
   * - ``"stream"``
     - CUDA stream handle
     - ``int64_t`` (cast to ``cudaStream_t``)
   * - ``"attr.<name>"``
     - Scalar attribute — **type auto-inferred** from C++ signature
     - Depends on C++ parameter type (see table below)
   * - ``"attr.<name>:<type>"``
     - Scalar attribute — **type explicit** in the token
     - Depends on ``<type>``

kernix Compatible Aliases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

kernix also accepts tokens from the **kernix** naming convention.  They
are transparently normalised before parsing, so both styles can be mixed freely.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - kernix token
     - Normalised to
     - Notes
   * - ``"args"``
     - ``"arg"``
     - Plural alias for a single input tensor
   * - ``"rets"``
     - ``"ret"``
     - Plural alias for a single output tensor
   * - ``"ctx.stream"``
     - ``"stream"``
     - Alternative CUDA stream token
   * - ``"attrs.<name>"``
     - ``"attr.<name>"``
     - Bare attribute (type still inferred from C++ signature)

Attribute Types
^^^^^^^^^^^^^^^

The ``attr`` token supports these types.  When using the bare
``"attr.<name>"`` form, kernix infers the type by parsing the C++ function
signature.  The explicit ``"attr.<name>:<type>"`` form is also accepted and
takes precedence.

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - Type String
     - C++ Type (impl param)
     - Python Value
     - XLA FFI scalar
   * - ``bool``
     - ``bool``
     - ``True`` / ``False``
     - ✓ native
   * - ``int8``
     - ``int8_t``
     - ``numpy.int8``
     - ✓ native
   * - ``uint8``
     - ``uint8_t``
     - ``numpy.uint8``
     - ✓ native
   * - ``int16``
     - ``int16_t``
     - ``numpy.int16``
     - ✓ native
   * - ``uint16``
     - ``uint16_t``
     - ``numpy.uint16``
     - ✓ native
   * - ``int32``
     - ``int32_t``
     - ``numpy.int32``
     - ✓ native
   * - ``uint32``
     - ``uint32_t``
     - ``numpy.uint32``
     - ✓ native
   * - ``int64``
     - ``int64_t``
     - ``numpy.int64``
     - ✓ native
   * - ``uint64``
     - ``uint64_t``
     - ``numpy.uint64``
     - ✓ native
   * - ``float32``
     - ``float``
     - ``numpy.float32``
     - ✓ native
   * - ``float64``
     - ``double``
     - ``numpy.float64``
     - ✓ native
   * - ``complex64``
     - ``std::complex<float>``
     - ``numpy.complex64``
     - ✓ native
   * - ``complex128``
     - ``std::complex<double>``
     - ``numpy.complex128``
     - ✓ native
   * - ``float16``
     - ``uint16_t`` (raw bits)
     - ``numpy.float16(x).view(numpy.uint16)``
     - ⚠ via uint16
   * - ``bfloat16``
     - ``uint16_t`` (raw bits)
     - ``bfloat16_val.view(numpy.uint16)``
     - ⚠ via uint16

.. note::

   **float16 / bfloat16 attrs**: XLA FFI has no native scalar attr decoding
   for 16-bit float types.  kernix maps them to ``uint16_t``; the C++ function
   receives the raw bit pattern and must reinterpret internally (e.g.
   ``__half h = *reinterpret_cast<const __half*>(&bits);``).  At call time,
   pass the raw bits: ``scale=numpy.float16(1.5).view(numpy.uint16)``.
   For most ML use cases, ``float32`` is preferable.

CUDA Header Convention
----------------------

kernix does **not** auto-inject ``<cuda_runtime.h>`` into your kernel source.
Always add it explicitly at the top so that ``cudaStream_t`` and other CUDA
runtime types are in scope:

.. code-block:: cpp

   #include <cuda_runtime.h>        // ← required for cudaStream_t, etc.
   #include "brainevent/common.h"   // BE::Tensor, BE_CUDA_CHECK, ...

   // @BE my_kernel arg ret stream
   void my_kernel(const BE::Tensor x, BE::Tensor out, int64_t stream) {
       auto s = (cudaStream_t)stream;
       // ...
   }

Examples
--------

Basic CUDA kernel (two inputs, one output, stream):

.. code-block:: python

   functions={"vector_add": ["arg", "arg", "ret", "stream"]}

Kernel with a scalar attribute — **bare form** (type inferred from C++):

.. code-block:: python

   functions={"scale_by": ["arg", "ret", "attr.scale_factor", "stream"]}

Kernel with a scalar attribute — **explicit form**:

.. code-block:: python

   functions={"scale_by": ["arg", "ret", "attr.scale_factor:float32", "stream"]}

Multiple outputs:

.. code-block:: python

   functions={"split": ["arg", "ret", "ret", "stream"]}

Multiple attributes:

.. code-block:: python

   functions={"scale_add": ["arg", "ret", "attr.scale", "attr.offset", "stream"]}

kernix style (equivalent to the above):

.. code-block:: python

   # These are all identical after normalisation:
   functions={"vector_add": ["args", "args", "rets", "ctx.stream"]}
   functions={"scale_by":   ["args", "rets", "attrs.scale_factor", "ctx.stream"]}

Function Signature Convention
------------------------------

Your C++ function parameters must follow this order:

1. Input tensors (``"arg"`` tokens) as ``const BE::Tensor``
2. Output tensors (``"ret"`` tokens) as ``BE::Tensor``
3. Scalar attributes (``"attr.*"`` tokens) as the corresponding C++ type
4. CUDA stream (``"stream"`` token) as ``int64_t``

.. code-block:: cpp

   // Matches: ["arg", "ret", "attr.scale", "stream"]
   void my_kernel(const BE::Tensor input,
                  BE::Tensor output,
                  float scale,
                  int64_t stream);

Attribute Type Inference
------------------------

When you write ``"attr.name"`` without a type suffix, kernix parses the C++
function signature to determine the type automatically.  The following C++
parameter types are recognised:

.. list-table::
   :header-rows: 1
   :widths: 35 25

   * - C++ Parameter Type
     - Inferred attr type
   * - ``bool``
     - ``bool``
   * - ``int8_t``, ``char``
     - ``int8``
   * - ``uint8_t``, ``unsigned char``
     - ``uint8``
   * - ``int16_t``, ``short``
     - ``int16``
   * - ``uint16_t``, ``unsigned short``
     - ``uint16``
   * - ``int32_t``, ``int``
     - ``int32``
   * - ``uint32_t``, ``unsigned int``
     - ``uint32``
   * - ``int64_t``, ``long long``
     - ``int64``
   * - ``uint64_t``, ``unsigned long long``
     - ``uint64``
   * - ``float``
     - ``float32``
   * - ``double``
     - ``float64``
   * - ``std::complex<float>``
     - ``complex64``
   * - ``std::complex<double>``
     - ``complex128``

Leading ``const`` qualifiers are stripped before lookup.  Pointer types,
``__half``, ``__nv_bfloat16``, and other non-standard types are **not**
auto-inferred — use the explicit ``"attr.name:<type>"`` form instead.

Auto-Detection (CPU)
--------------------

For CPU functions, you can pass a **list** of function names instead of a dict.
kernix will parse the C++ signatures and infer the full arg_spec automatically:

- ``const BE::Tensor`` parameters become ``"arg"``
- Non-const ``BE::Tensor`` parameters become ``"ret"``
- Scalar parameters become ``"attr.<name>:<type>"``

.. code-block:: python

   # These are equivalent:
   functions=["add_one"]
   functions={"add_one": ["arg", "ret"]}

.. code-block:: cpp

   // Auto-detected as ["arg", "ret"]
   void add_one(const BE::Tensor x, BE::Tensor y);

Passing Attributes at Call Time
-------------------------------

Scalar attributes are passed as keyword arguments to the **returned callable**
from ``jax.ffi.ffi_call()``:

.. code-block:: python

   # CORRECT
   jax.ffi.ffi_call("target", spec)(x, scale_factor=np.float32(3.0))

   # WRONG -- attributes must NOT go to ffi_call() directly
   jax.ffi.ffi_call("target", spec, scale_factor=...)(x)
