Apply event-driven synaptic plasticity
======================================

Plasticity rules update synaptic weights in response to spikes. ``brainevent`` provides
event-driven plasticity operators that touch only the weights connected to neurons that
actually fired — the same sparsity principle as the matrix products. This guide shows when
to reach for each operator; the full, runnable derivation is in
:doc:`/tutorials/data-structures/05_synaptic_plasticity`.


Pre- vs post-synaptic updates
------------------------------

Plastic updates are driven either by **pre-synaptic** spikes (the source neuron fired) or
**post-synaptic** spikes (the target neuron fired). ``brainevent`` ships both directions for
each storage format:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Storage
     - Pre-synaptic driven
     - Post-synaptic driven
   * - **CSR** (sparse)
     - :func:`~brainevent.update_csr_on_binary_pre`
     - :func:`~brainevent.update_csr_on_binary_post`
   * - **Dense**
     - :func:`~brainevent.update_dense_on_binary_pre`
     - :func:`~brainevent.update_dense_on_binary_post`


Choosing CSR vs dense
---------------------

- Use the **CSR** operators when connectivity is sparse and fixed (the common case for
  large networks) — only the stored synapses are visited.
- Use the **dense** operators for small, fully-connected layers where a dense weight matrix
  is already in play.

In both cases the spike trigger is a :class:`~brainevent.BinaryArray`, so only active events
contribute to the update — the cost scales with the number of spikes, not the number of
synapses.

.. seealso::

   The exact operator signatures (weights, spike vector, learning-rate arguments, and the
   associated ``*_p`` primitives) are listed under **Plasticity operations** in
   :doc:`/reference/apis/operations`.
