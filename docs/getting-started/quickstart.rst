Quickstart
==========

This page gets you from zero to a working event-driven matrix multiplication in a couple of
minutes. For the *why* behind it, see :doc:`/explanation/event-driven-computation`.


The core idea
-------------

The brain computes with **spikes** — sparse, binary events. ``brainevent`` exploits that
sparsity: wrap a spike vector in :class:`~brainevent.BinaryArray`, and any matrix
multiplication against it skips the zeros and processes only the neurons that fired.

.. code-block:: python

   import brainevent
   import jax.numpy as jnp

   # 1 = spike, 0 = no spike
   spikes = brainevent.BinaryArray(jnp.array([1, 0, 1, 0, 1], dtype=jnp.float32))


Multiply by a connectivity matrix
----------------------------------

A ``BinaryArray`` multiplies against dense arrays *and* any of ``brainevent``'s sparse
connectivity structures. The operation is event-driven in every case:

.. code-block:: python

   import jax.numpy as jnp

   # Dense weights (jax/numpy array)
   weights = jnp.ones((5, 3))
   out_dense = spikes @ weights

   # CSR sparse matrix — one synapse per pre-synaptic neuron
   data = jnp.arange(1, 6, dtype=jnp.float32)
   indices = jnp.array([0, 1, 2, 0, 1])
   indptr = jnp.arange(6)
   csr = brainevent.CSR((data, indices, indptr), shape=(5, 3))
   out_csr = spikes @ csr

   # Just-in-time connectivity — never materialises the full matrix
   jitc = brainevent.JITCScalarR(0.2, 0.5, 0, shape=(5, 3))   # weight, prob, seed
   out_jitc = spikes @ jitc

   # Fixed fan-out — two post-synaptic targets per pre-synaptic neuron
   conn = jnp.array([[0, 1], [1, 2], [0, 2], [1, 0], [2, 1]])
   fixed = brainevent.FixedNumPerPre(jnp.full((5, 2), 0.5), conn, shape=(5, 3))
   out_fixed = spikes @ fixed


Works inside JAX transformations
---------------------------------

Everything composes with ``jax.jit``, ``jax.grad``, and ``jax.vmap``:

.. code-block:: python

   import jax

   @jax.jit
   def step(spikes, csr):
       return spikes @ csr

   out = step(spikes, csr)


Next steps
----------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Learn connectivity data
      :link: /tutorials/data-structures/index
      :link-type: doc

      Compare explicit, fixed-count, and generated connectivity structures.

   .. grid-item-card:: Learn event-driven operations
      :link: /tutorials/events/index
      :link-type: doc

      Work from binary event representation to event-triggered updates.

   .. grid-item-card:: Solve a specific task
      :link: /how-to/data-structures/index
      :link-type: doc

      Jump to a how-to recipe.

   .. grid-item-card:: Understand the model
      :link: /explanation/event-driven-computation
      :link-type: doc

      Read the conceptual background.

   .. grid-item-card:: Look up an API
      :link: /reference/apis/index
      :link-type: doc

      Browse the reference.
