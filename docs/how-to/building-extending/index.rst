Building & extending
====================

Task-oriented recipes for building event-driven networks and extending ``brainevent``
with your own kernels. For the concepts behind custom kernels, see
:doc:`/explanation/custom-kernel-architecture`.

Choose the network recipe when you want to assemble event-driven communication into a
spiking model, or the compilation recipe when you need an operation implemented in
native source. The network guide introduces the pattern of sending activity through
excitatory and inhibitory connectivity and points to complete COBA and CUBA examples.
The compilation guide shows how to make C++ or CUDA code callable from JAX through the
provided loading interface. Before adapting either recipe, identify the shapes of the
arrays entering your computation and the result you expect to produce. For native
kernels, also check the compiler and device requirements described in the guide. Use the
linked architecture explanation to understand the compilation bridge, and consult the
kernel reference when you need details about argument specifications, compiler options,
or cached build artifacts.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Build a network
      :link: build-coba-cuba-network
      :link-type: doc

      Connect event-driven communication to COBA and CUBA network examples.

   .. grid-item-card:: Compile a native kernel
      :link: compile-raw-cuda-cpp
      :link-type: doc

      Load C++ or CUDA source and call the resulting kernel from JAX.

.. toctree::
   :maxdepth: 1
   :caption: Building & extending

   build-coba-cuba-network
   compile-raw-cuda-cpp
