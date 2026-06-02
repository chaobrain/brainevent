Tutorials
=========

Learning-oriented, hands-on notebooks. Each one is a guided lesson you can run top to
bottom; together they build up from the basic event array to writing your own
high-performance kernels.

If you are brand new, start with :doc:`/getting-started/quickstart`, then work through the
**Data structures & operators** track in order.


Data structures & operators
----------------------------

The event representations ``brainevent`` provides and the operators that act on them.

.. toctree::
   :maxdepth: 1
   :caption: Data structures & operators

   data-structures/01_eventarray_basics
   data-structures/02_sparse_matrices
   data-structures/03_jit_connectivity
   data-structures/04_fixed_connections
   data-structures/05_synaptic_plasticity


Custom operators
----------------

Extend ``brainevent`` with your own kernels — from high-level Numba/Warp decorators down to
hand-written C++ and CUDA.

.. toctree::
   :maxdepth: 1
   :caption: Custom operators

   custom-operators/01_numba
   custom-operators/02_numba_cuda
   custom-operators/03_warp
   custom-operators/04_cpp
   custom-operators/05_cuda
