Use JIT connectivity for large networks
=======================================

Just-in-time connectivity (**JITC**) represents a random connectivity matrix by its
*generator* — a probability, a weight distribution, and a seed — instead of by its entries.
The matrix is never stored; each row (or column) is regenerated on the fly inside the
kernel. This makes the memory cost independent of the number of synapses, which is what lets
you scale to networks with millions of neurons. The theory is covered in
:doc:`/explanation/jit-connectivity`.


Scalar (homogeneous) weights
----------------------------

When every connection shares one weight, use :class:`~brainevent.JITCScalarR` (row variant)
or :class:`~brainevent.JITCScalarC` (column variant):

.. code-block:: python

   import brainevent
   import jax.numpy as jnp

   conn = brainevent.JITCScalarR(
       (0.5, 0.01, 0),          # (shared weight, connection probability, seed)
       shape=(100_000, 100_000),
   )

   spikes = brainevent.BinaryArray(jnp.zeros(100_000).at[::1000].set(1.0))
   currents = spikes @ conn        # 100k x 100k, never materialised


Distributed weights
--------------------

For heterogeneous weights, use the normally- or uniformly-distributed variants
(``JITCNormalR``/``JITCNormalC`` and ``JITCUniformR``/``JITCUniformC``). They take
distribution parameters in place of the single ``weight`` argument — see
:doc:`/reference/apis/sparsedata` for the exact constructor signatures.


Reproducibility
---------------

The ``seed`` fully determines the realised connectivity. The **same seed always yields the
same matrix**, across processes and devices — so a JITC matrix is reproducible without being
stored. Use distinct seeds for statistically independent populations:

.. code-block:: python

   exc = brainevent.JITCScalarR((1.6, 0.02, 1), shape=(n, n))
   inh = brainevent.JITCScalarR((-9.0, 0.02, 2), shape=(n, n))


Choosing the event-driven side
------------------------------

A JITC matrix regenerates its connections along one direction, and only products along
that direction skip inactive events. Pick it from the product your model computes:

=========================================  ===================================
your product                               construct with
=========================================  ===================================
``spikes @ W`` (``W`` is ``[pre, post]``)  ``event_side='left'`` (the default)
``W @ spikes`` (``W`` is ``[post, pre]``)  ``event_side='right'``
dense ``W @ x``                            ``event_side='left'`` (the default)
dense ``x @ W`` (rate models)              ``event_side='right'``
=========================================  ===================================

.. code-block:: python

   W = brainevent.JITCScalarR((0.5, 0.01, 0), shape=(n_post, n_pre), event_side='right')
   currents = W @ spikes          # event-driven: only active presynaptic columns are walked

``event_side`` is the readable name of ``corder`` (``'left'`` is ``corder=True``). The two
settings draw different connectivity with the same statistics, so fix one per matrix; ``M.T``
flips it, so a matrix and its transpose describe the same connections. For
``spikes @ W`` the default is also fast when training: the input gradient is the dense
``W @ ct``, which runs on the fast gather.

A product on the other side still gives the right result, but walks the whole matrix; the
first such call emits a :class:`~brainevent.PerformanceWarning`. If you need the *same*
connectivity fast in both directions and it fits in memory, materialise it once with
``W.tocsr()``. A :class:`~brainevent.CSR` matrix runs its other direction as a gather over
the stored indices instead of regenerating the matrix, and on GPU with the ``cuda_raw``
backend both directions are event-driven once its transpose layout is cached. The price is
``O(nnz)`` memory: 8 bytes per connection (800 MB for ``100k x 100k`` at ``p = 0.01``),
about twice that with the transpose layout.

.. note::

   JITC weights cannot be inspected or learned individually — the matrix only exists
   transiently inside the kernel. If you need plastic or addressable weights, use
   :class:`~brainevent.CSR` instead (see :doc:`synaptic-plasticity`).
