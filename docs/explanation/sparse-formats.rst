Sparse format trade-offs
========================

``brainevent`` provides several connectivity representations because no single format is best
for every situation. This page explains the trade-offs so you can reason about the choice;
for a quick decision table, see :doc:`/how-to/data-structures/choosing-a-sparse-format`.


Coordinate (COO) input
----------------------

The most intuitive way to describe a sparse matrix is as a list of ``(row, column, value)``
triplets — the coordinate, or **COO**, form. It is convenient to *build* but inefficient to
*multiply*, because the entries are unordered. ``brainevent`` therefore treats COO as an
input format: the :func:`~brainevent.coo2csr` helper converts coordinate triplets into a CSR
matrix ready for event-driven products. Related index converters
(:func:`~brainevent.csr_to_coo_index`, :func:`~brainevent.csr_to_csc_index`,
:func:`~brainevent.coo_to_csc_index`) move between layouts.


CSR vs CSC
----------

:class:`~brainevent.CSR` (Compressed Sparse Row) stores, for each row, the column indices and
values of its non-zeros. This makes **row-oriented** access cheap — ideal when a spike vector
selects rows. :class:`~brainevent.CSC` is the transpose idea: column-compressed, so
**column-oriented** access is cheap.

The right choice depends on which side of the product the spike vector sits and which
dimension it indexes. Picking the layout whose compressed dimension aligns with the
contraction keeps memory access contiguous and the kernel fast. The two formats store the
same information and convert into one another, so the decision is purely about performance.


Explicit storage vs generation
-------------------------------

CSR/CSC store every non-zero explicitly. That is the right model when the connectivity is
**fixed and inspectable** — you can read, slice, and learn individual weights. The memory
cost is proportional to the number of synapses.

For **random** connectivity at large scale, explicit storage becomes the bottleneck: a
million-neuron network at 1 % density has ten billion synapses. Just-in-time connectivity
(JITC) sidesteps this by regenerating connectivity from a seed inside the kernel, so memory
no longer scales with synapse count — at the cost of losing per-weight addressability. This
trade-off is the subject of :doc:`jit-connectivity`.


Fixed-degree structure
----------------------

Many biological networks have a **fixed number** of connections per neuron rather than a
fixed probability. :class:`~brainevent.FixedNumPerPre` and
:class:`~brainevent.FixedNumPerPost` encode exactly that, which is both more biologically
faithful and more memory-predictable than a probabilistic format when the degree is known.


Matrices under ``jax.jit`` and ``jax.grad``
--------------------------------------------

CSR, CSC and the fixed-degree formats are JAX pytrees whose leaves are the values
(``data``) *and* the structure (``indices``, ``indptr``). The structure is therefore an
ordinary input of a jitted function: it is not baked into the compiled program, and two
matrices with the same shapes share one compilation.

* **Gradients** flow through the values. Differentiate ``M.data`` and rebuild the matrix
  with ``M.with_data(d)``; ``jax.grad(f, allow_int=True)(M)`` also works and returns
  ``float0`` cotangents for the index leaves.
* **Mirror layout.** The unfavorable product direction (and the matching plasticity
  update) is event-driven through a *mirror*: the other compression of the same structure.
  It is built once from a concrete structure and shared by ``with_data``, ``apply``,
  arithmetic and ``.T``. A matrix that enters a jitted function as an argument has a traced
  structure, so call ``M = M.build_weight_indices()`` first; otherwise that direction runs
  on a slower mirror-free path and warns once with :class:`~brainevent.PerformanceWarning`.
  A matrix closed over by the jitted function needs nothing.
* **Structure-changing operations** plan the new structure on the host:
  ``diag_add`` needs ``M = M.prepare_diag_add()`` before ``jax.jit`` when ``M`` is an
  argument; ``slice_rows`` on CSR/CSC needs concrete row indices.


Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 24

   * - Format
     - Memory
     - Per-weight access
     - Best for
   * - Dense
     - O(rows × cols)
     - Yes
     - small / truly dense
   * - CSR / CSC
     - O(non-zeros)
     - Yes
     - fixed explicit sparsity
   * - JITC
     - O(1) in synapses
     - No
     - large random connectivity
   * - Fixed fan-in/out
     - O(neurons × degree)
     - structural
     - fixed-degree biology
