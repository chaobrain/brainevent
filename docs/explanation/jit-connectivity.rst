Just-in-time connectivity
=========================

Just-in-time connectivity (**JITC**) is ``brainevent``'s answer to a hard scaling problem:
how do you multiply a spike vector by a random connectivity matrix that is too large to
store? For the practical recipe, see :doc:`/how-to/jit-connectivity-large-networks`.


The memory problem
------------------

A randomly connected network of :math:`N` neurons with connection probability :math:`p` has
on the order of :math:`pN^2` synapses. For :math:`N = 10^6` and :math:`p = 0.01`, that is
:math:`10^{10}` synapses — far beyond device memory if stored explicitly, even in a
compressed sparse format. Yet the *information* defining that matrix is tiny: a probability,
a weight (or weight distribution), and a random seed.


The generator idea
------------------

JITC stores only the **generator**, never the matrix. Each time a row (or column) is needed
during a matrix–vector product, the kernel uses a deterministic, seedable random number
generator to regenerate that row's connections and weights on the fly, uses them, and
discards them. Nothing is materialised.

Two properties make this sound:

- **Determinism** — the same seed always reproduces the same connections, so the matrix is
  well-defined and reproducible across processes and devices despite never being stored.
- **Locality** — each row can be regenerated independently, which maps naturally onto the
  parallel, event-driven kernels: only rows touched by an active spike are ever generated.

The result is a matrix product whose **memory cost is independent of the number of
synapses** and whose compute cost still scales with the number of spikes.


Weight distributions
--------------------

The realised weights come from a distribution chosen per variant:

- **Scalar** (:class:`~brainevent.JITCScalarR` / ``…C``) — every connection shares one
  constant weight.
- **Normal** (``JITCNormalR`` / ``…C``) — weights drawn from a Gaussian.
- **Uniform** (``JITCUniformR`` / ``…C``) — weights drawn uniformly from an interval.

In every case the weights are regenerated from the seed, never stored.


The trade-off
-------------

JITC buys constant memory at the price of **addressability**: because the matrix exists only
transiently inside the kernel, you cannot inspect, slice, or learn individual weights. If a
model needs plastic or addressable synapses, an explicit :class:`~brainevent.CSR` matrix is
the right tool instead (see :doc:`sparse-formats`).

.. note::

   The row- (``R``) and column- (``C``) oriented variants differ only in which dimension is
   regenerated contiguously. As with CSR/CSC, choose the one that aligns with your spike
   vector's orientation for best performance.
