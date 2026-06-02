Choose a connectivity format
============================

``brainevent`` offers several ways to represent the connectivity a spike vector multiplies
against. This guide helps you pick the right one. For the reasoning behind the formats, see
:doc:`/explanation/sparse-formats`.


Decision guide
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Format
     - Use when
     - Avoid when
   * - **Dense** (``jax``/``numpy`` array)
     - The matrix is small or genuinely dense (>~25 % non-zero), or you need arbitrary
       per-entry weights with the simplest possible code.
     - The matrix is large and sparse — you waste memory and compute on zeros.
   * - :class:`~brainevent.CSR` / :class:`~brainevent.CSC`
     - You have an explicit, fixed sparse matrix and want fast row- (CSR) or column- (CSC)
       oriented event-driven products.
     - Connectivity is generated randomly and the full matrix would not fit in memory.
   * - **JITC** (:class:`~brainevent.JITCScalarR`, ``JITCNormalR``, ``JITCUniformR``, …)
     - Connectivity is random with a fixed probability, and you want to **never materialise**
       the matrix — it is regenerated on the fly from a seed.
     - You need to inspect, mutate, or learn individual weights.
   * - **Fixed fan-in/out** (:class:`~brainevent.FixedPreNumConn`,
       :class:`~brainevent.FixedPostNumConn`)
     - Each neuron has a fixed *number* of connections (biologically common), and you want
       that structure encoded directly.
     - Connection counts vary per neuron, or you need an explicit weight matrix.


Rule of thumb
-------------

- **Explicit and reusable** → ``CSR``/``CSC``.
- **Random and huge** → ``JITC*`` (memory cost is independent of density; see
  :doc:`jit-connectivity-large-networks`).
- **Fixed degree per neuron** → ``FixedPreNumConn`` / ``FixedPostNumConn``.

Every format multiplies the same way against a :class:`~brainevent.BinaryArray`, so you can
swap formats without changing the surrounding code:

.. code-block:: python

   import brainevent
   import jax.numpy as jnp

   spikes = brainevent.BinaryArray(jnp.array([1, 0, 1, 0, 1], dtype=jnp.float32))

   for conn in (csr, jitc, fixed):       # any format
       out = spikes @ conn               # identical call site


Row vs column variants (``R`` / ``C``)
---------------------------------------

JITC and CSR/CSC come in row- and column-oriented variants. Choose based on which dimension
your spike vector indexes — the variant that keeps the contraction along contiguous memory
is faster. When in doubt, benchmark both with
:func:`~brainevent.benchmark_function`.

.. seealso::

   :doc:`/tutorials/data-structures/02_sparse_matrices` and
   :doc:`/tutorials/data-structures/03_jit_connectivity` walk through each format
   interactively.
