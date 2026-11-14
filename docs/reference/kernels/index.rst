Custom kernels
==============

Information-oriented reference for the raw C++/CUDA kernel toolchain (see
:doc:`/explanation/custom-kernel-architecture` for the overview).

Use this reference while implementing or maintaining a native kernel that is loaded
through brainevent. Begin with the argument specification page to understand how Python-
side inputs and outputs are described to the wrapper generator. Consult the C++ API for
the tensor types and helper interfaces available inside your source. The compiler
options page explains settings you can adjust when building CUDA code, and the caching
page describes how compiled artifacts are identified and reused. These pages are most
useful alongside a working example from the custom operator tutorials or the raw
compilation how-to guide. Before changing a build setting, check which part of the
compilation it affects and whether it changes the resulting cache entry. When diagnosing
a problem, distinguish the argument interface, source implementation, compiler
configuration, and artifact reuse so you can consult the relevant reference first.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Argument specifications
      :link: arg-spec
      :link-type: doc

      Describe kernel parameters for generated XLA FFI wrappers.

   .. grid-item-card:: C++ API
      :link: cpp-api
      :link-type: doc

      Look up tensor types and helpers available to native kernels.

   .. grid-item-card:: Compiler options
      :link: compiler-options
      :link-type: doc

      Inspect CUDA compilation and optimization settings.

   .. grid-item-card:: Caching
      :link: caching
      :link-type: doc

      Understand cache keys, artifact storage, and cache configuration.

.. toctree::
   :maxdepth: 1
   :caption: Custom kernels

   arg-spec
   cpp-api
   compiler-options
   caching
