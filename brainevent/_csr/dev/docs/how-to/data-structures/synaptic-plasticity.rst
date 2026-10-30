Apply event-driven synaptic plasticity
======================================

Plasticity rules update synaptic weights in response to spikes. ``brainevent`` provides
event-driven plasticity operators that touch only the weights connected to neurons that
actually fired — the same sparsity principle as the matrix products. This guide shows when
to reach for each operator; the full, runnable derivation is in
:doc:`/tutorials/events/synaptic-plasticity`.


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

Differentiating through plasticity
----------------------------------

The binary operators above are differentiable: ``jax.grad`` and ``jax.jvp`` work with
respect to the weights, the trace, and the clipping bounds ``w_min`` / ``w_max``. Spikes are
treated as non-differentiable. This lets weights evolve by plasticity *and* receive
gradients in the same training step.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import brainevent

    csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.], [0.3, 0.]]))
    pre_spike = jnp.array([True, False, True])

    def loss(w, post_trace):
        w = brainevent.update_csr_on_binary_pre(
            w, csr.indices, csr.indptr, pre_spike, post_trace, 0., 5., shape=csr.shape)
        return w.sum()

    g_w, g_trace = jax.grad(loss, argnums=(0, 1))(csr.data, jnp.array([0.1, 0.2]))


Custom per-edge rules
---------------------

The binary operators implement the additive rule ``w += trace``. For anything else — a
weight-dependent (multiplicative) update, several state variables per synapse such as an
eligibility trace, or a triplet rule — use :func:`~brainevent.csr_apply_on_pre` /
:func:`~brainevent.csr_apply_on_post` (or the :meth:`~brainevent.CSR.apply_on_pre` /
:meth:`~brainevent.CSR.apply_on_post` methods). You write the rule for **one** synapse; it
is applied to every synapse of a firing neuron, and the other synapses are left unchanged:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import brainevent

    csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.], [0.3, 0.]]))
    state = {'w': csr.data, 'elig': jnp.zeros_like(csr.data)}

    def stdp(edge, pre, post, p):
        # eligibility: decays, potentiated by the post trace, depressed by the pre trace
        elig = p['decay'] * edge['elig'] + post - pre
        # soft-bounded (weight-dependent) update towards w_max
        w = edge['w'] + p['lr'] * (p['w_max'] - edge['w']) * elig
        return {'w': w, 'elig': elig}

    params = {'lr': 0.01, 'w_max': 2.0, 'decay': 0.9}
    pre_trace = jnp.array([0.2, 0.0, 0.5])
    post_trace = jnp.array([0.4, 0.1])
    pre_spike = jnp.array([True, False, True])

    new_state = brainevent.csr_apply_on_pre(
        stdp, state, csr.indices, csr.indptr, pre_spike,
        pre_vals=pre_trace, post_vals=post_trace, params=params, shape=csr.shape,
    )

    # Gradients flow through the rule, so plasticity parameters can be learned.
    def loss(lr):
        new = csr.apply_on_pre(
            stdp, pre_spike, edge_state=state,
            pre_vals=pre_trace, post_vals=post_trace, params=dict(params, lr=lr),
        )
        return jnp.sum((new['w'] - 1.0) ** 2)

    d_lr = jax.grad(loss)(0.01)

``edge_state`` is any pytree of ``(nse,)`` arrays in CSR data order, ``pre_vals`` /
``post_vals`` are per-neuron pytrees gathered to each synapse, and ``params`` is passed to
every call. The rule is evaluated on all synapses and masked, so its cost scales with the
number of synapses; for the plain additive rule the binary operators above are faster.

.. seealso::

   The exact operator signatures (weights, spike vector, learning-rate arguments, and the
   associated ``*_p`` primitives) are listed under **Plasticity operations** in
   :doc:`/reference/apis/operations`.
