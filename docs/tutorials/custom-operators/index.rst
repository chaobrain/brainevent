Custom operators
================

Learning-oriented, hands-on notebooks. Each one is a guided lesson you can run top to
bottom; together they build up from the basic event array to writing your own
high-performance kernels.

If you are brand new, start with :doc:`/getting-started/quickstart`.


Extend ``brainevent`` with your own kernels — from high-level Numba/Warp decorators down to
hand-written C++ and CUDA.

Choose a notebook according to the language and execution device you want to work with.
The Numba lesson provides a starting point for CPU kernels, while the Numba CUDA and
Warp lessons introduce routes for GPU implementations. The C++ and CUDA lessons cover
writing native source and connecting the compiled result to JAX. Within each lesson,
follow the examples in sequence to understand the inputs, outputs, and registration
steps before adapting the implementation. Check that your environment has the
dependencies and hardware required by the selected backend. You can use the higher-level
lessons as an introduction before moving to native code, or go directly to the backend
you already know. For parameter details and compilation settings encountered along the
way, keep the custom kernel reference available as a companion.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Numba on CPU
      :link: 01_numba
      :link-type: doc

      Build a custom CPU operator using Numba.

   .. grid-item-card:: Numba CUDA
      :link: 02_numba_cuda
      :link-type: doc

      Write a GPU operator with Numba CUDA.

   .. grid-item-card:: Warp
      :link: 03_warp
      :link-type: doc

      Explore custom GPU kernels written with Warp.

   .. grid-item-card:: C++
      :link: 04_cpp
      :link-type: doc

      Compile native CPU code and call it from JAX.

   .. grid-item-card:: CUDA
      :link: 05_cuda
      :link-type: doc

      Connect a native CUDA kernel to JAX computations.

.. toctree::
   :maxdepth: 1
   :caption: Custom operators

   01_numba
   02_numba_cuda
   03_warp
   04_cpp
   05_cuda
