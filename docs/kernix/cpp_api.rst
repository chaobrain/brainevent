C++ API
=======

User kernels include ``"brainevent/common.h"`` which provides the core C++
types and macros.  Internal headers (``ffi_compat.h``, ``dtypes.h``) are
auto-injected by the wrapper generator and should **not** be included manually.

BE::Tensor
----------

A lightweight, non-owning view over a contiguous tensor buffer.  Stores shape
and C-contiguous strides internally (up to 8 dimensions) so the object is
trivially copyable and can be passed by value into CUDA kernel argument lists.

.. code-block:: cpp

   #include "brainevent/common.h"

   class BE::Tensor {
   public:
       // Data access
       void* data_ptr() const noexcept;                       // untyped
       template <typename T> T* data_ptr() const noexcept;   // typed overload
       void* data() const noexcept;                           // alias for data_ptr()

       // Shape
       int ndim() const noexcept;
       int64_t size(int i) const noexcept;    // size along dimension i
       int64_t shape(int i) const noexcept;   // alias for size(i)
       int64_t stride(int i) const noexcept;
       const int64_t* shape_ptr() const noexcept;
       const int64_t* strides_ptr() const noexcept;

       // Dtype
       DType dtype() const noexcept;
       size_t element_size() const noexcept;

       // Aggregate queries
       int64_t numel() const noexcept;
       size_t nbytes() const noexcept;
       bool is_contiguous() const noexcept;
   };

BE::DType
---------

Enum class mirroring JAX / NumPy dtypes:

.. code-block:: cpp

   enum class DType : uint8_t {
       Float16    = 0,
       Float32    = 1,
       Float64    = 2,
       BFloat16   = 3,
       Int8       = 4,
       Int16      = 5,
       Int32      = 6,
       Int64      = 7,
       UInt8      = 8,
       UInt16     = 9,
       UInt32     = 10,
       UInt64     = 11,
       Bool       = 12,
       Complex64  = 13,
       Complex128 = 14,
       Invalid    = 255,
   };

Utility functions:

- ``dtype_size(DType dt) -> size_t`` — byte width of one element
- ``dtype_name(DType dt) -> const char*`` — human-readable name (e.g. ``"float32"``)

Error Checking Macros
---------------------

Defined in ``brainevent/check.h`` (included by ``brainevent/common.h``).

``BE_CHECK``
^^^^^^^^^^^^

Runtime assertion with a streaming error message:

.. code-block:: cpp

   BE_CHECK(idx >= 0 && idx < n) << "Index out of range: " << idx;

Aborts with a descriptive message if the condition is false.

``BE_CUDA_CHECK``
^^^^^^^^^^^^^^^^^

Check CUDA API return codes:

.. code-block:: cpp

   BE_CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));

Also useful after kernel launches via the convenience macro:

.. code-block:: cpp

   my_kernel<<<blocks, threads, 0, stream>>>(args...);
   BE_CHECK_KERNEL_LAUNCH();   // expands to BE_CUDA_CHECK(cudaGetLastError())

Dispatch Macros
---------------

Defined in ``brainevent/dispatch.h``.  Include it explicitly in user code
if needed (it is auto-included in the generated FFI wrappers).

.. code-block:: cpp

   #include "brainevent/dispatch.h"

``BE_DISPATCH_FLOATING``
^^^^^^^^^^^^^^^^^^^^^^^^

Dispatch over floating-point types (float32, float64):

.. code-block:: cpp

   BE_DISPATCH_FLOATING(tensor.dtype(), scalar_t, {
       my_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
           static_cast<const scalar_t*>(tensor.data_ptr()),
           static_cast<scalar_t*>(out.data_ptr()),
           n);
   });

``BE_DISPATCH_INTEGRAL``
^^^^^^^^^^^^^^^^^^^^^^^^

Dispatch over integer types (int8–int64, uint8–uint64).

``BE_DISPATCH_ALL_TYPES``
^^^^^^^^^^^^^^^^^^^^^^^^^

Dispatch over all numeric types (floating + integral).

Complete Example
----------------

.. code-block:: cpp

   #include <cuda_runtime.h>
   #include "brainevent/common.h"

   __global__ void scale_kernel(const float* x, float* out, int n, float factor) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) out[idx] = x[idx] * factor;
   }

   // @BE scale_by arg ret attr.scale_factor:float32 stream
   void scale_by(const BE::Tensor x, BE::Tensor out,
                 float scale_factor, int64_t stream) {
       int n = x.numel();
       scale_kernel<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
           static_cast<const float*>(x.data_ptr()),
           static_cast<float*>(out.data_ptr()),
           n, scale_factor);
       BE_CHECK_KERNEL_LAUNCH();
   }
