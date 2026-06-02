Build an event-driven E/I network
==================================

``brainevent`` provides the event-driven *communication* layer of a spiking network: spikes
fan out through a sparse connectivity structure to deliver synaptic currents. This guide
shows the core pattern, then points to complete, runnable COBA and CUBA implementations of
the classic Vogels–Abbott balanced network.


The core pattern
----------------

A population emits a binary spike vector each step; multiplying it by a connectivity
structure delivers currents to the post-synaptic population. Splitting the population into
excitatory and inhibitory groups with opposite-sign weights gives a balanced network:

.. code-block:: python

   import brainevent
   import brainunit as u
   import jax.numpy as jnp

   n_exc, n_inh = 3200, 800
   num = n_exc + n_inh

   # Fixed fan-out connectivity, excitatory (+) and inhibitory (-)
   exc_conn = brainevent.FixedPostNumConn(num_pre=n_exc, num_post=num,
                                          conn_num=80, weight=0.6 * u.mS, seed=1)
   inh_conn = brainevent.FixedPostNumConn(num_pre=n_inh, num_post=num,
                                          conn_num=80, weight=-6.7 * u.mS, seed=2)

   def synaptic_input(spikes):
       spk = brainevent.BinaryArray(spikes)
       g = spk[:n_exc] @ exc_conn + spk[n_exc:] @ inh_conn   # event-driven, unit-aware
       return g

Because the spike vector is a :class:`~brainevent.BinaryArray`, only neurons that fired this
step contribute work — the cost scales with the spike count, not the network size. Swap
``FixedPostNumConn`` for a :class:`~brainevent.JITCScalarR` to scale to networks too large to
store explicitly (see :doc:`jit-connectivity-large-networks`).


Full models (COBA & CUBA)
-------------------------

Complete networks combine this communication layer with neuron and synapse dynamics from the
`BrainX <https://brainx.chaobrain.com/>`_ ecosystem (``brainstate``, ``brainpy``,
``braintools``). Two reference implementations of the Vogels–Abbott balanced network live in
the repository's ``examples/`` directory:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Synapse model
   * - `examples/CUBA_2005.py <https://github.com/chaobrain/brainevent/blob/main/examples/CUBA_2005.py>`_
     - Current-based (CUBA) synapses.
   * - `examples/COBA_2005.py <https://github.com/chaobrain/brainevent/blob/main/examples/COBA_2005.py>`_
     - Conductance-based (COBA) synapses.

Both use ``brainevent``'s event-driven connectivity under the hood and scale from a few
thousand to hundreds of thousands of neurons by changing a single ``scale`` factor.

.. note::

   The reference models implement:

   - Vogels, T. P. & Abbott, L. F. (2005). *Signal propagation and logic gating in networks
     of integrate-and-fire neurons.* J. Neurosci., 25(46), 10786–95.
   - Brette, R. et al. (2007). *Simulation of networks of spiking neurons: a review of tools
     and strategies.* J. Comput. Neurosci., 23(3), 349–98.
