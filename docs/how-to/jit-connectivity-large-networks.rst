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
       num_pre=100_000,
       num_post=100_000,
       prob=0.01,       # connection probability
       weight=0.5,      # shared synaptic weight
       seed=0,          # fixes the realised connectivity
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

   exc = brainevent.JITCScalarR(num_pre=n, num_post=n, prob=0.02, weight=1.6, seed=1)
   inh = brainevent.JITCScalarR(num_pre=n, num_post=n, prob=0.02, weight=-9.0, seed=2)


Choosing ``R`` vs ``C``
-----------------------

The ``R`` variants are row-oriented and the ``C`` variants column-oriented. Pick the one
whose orientation matches how your spike vector is laid out so the contraction runs along
contiguous memory. If unsure, profile both with :func:`~brainevent.benchmark_function`.

.. note::

   JITC weights cannot be inspected or learned individually — the matrix only exists
   transiently inside the kernel. If you need plastic or addressable weights, use
   :class:`~brainevent.CSR` instead (see :doc:`synaptic-plasticity`).
