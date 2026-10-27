Just-in-time connectivity
=========================

Just-in-time connectivity (**JITC**) is ``brainevent``'s answer to a hard scaling problem:
how do you multiply a spike vector by a random connectivity matrix that is too large to
store? For the practical recipe, see :doc:`/how-to/data-structures/jit-connectivity-large-networks`.


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


Which side is event-driven
--------------------------

The random streams are regenerated per row (``corder=True``, the default) or per column
(``corder=False``). A product that runs along the streams scatters each active input and
skips the rest; a product that runs across them gathers by walking every stream:

=====================  ==============================  ==============================
product                ``corder=True`` (default)       ``corder=False``
=====================  ==============================  ==============================
``events @ M``         event-driven scatter            gather over every column
``M @ events``         gather over every row           event-driven scatter
dense ``x @ M``        scatter (atomic adds)           gather
dense ``M @ x``        gather                          scatter (atomic adds)
=====================  ==============================  ==============================

The two settings draw different matrices. ``M.T`` flips the flag, so a matrix and its
transpose always describe the same connectivity, and every product, ``todense()``,
``tocsr()`` and gradient of one matrix agree with one another. The ``R`` and ``C`` classes
are each other's transpose (``JITCScalarR(...).T`` is a ``JITCScalarC``); use ``corder``,
or its readable alias ``event_side`` (``'left'``: ``events @ M``, ``'right'``:
``M @ events``), not the class, to choose the event-driven side.

No setting can make both sides event-driven for one matrix: a stream lists the columns of
its row, and finding the rows that reach a given column means walking every stream. That is
why the slow side costs about as much as generating the whole matrix, whatever the event
rate.


Exact density
-------------

Each stream jumps ahead by random skips drawn uniformly from ``1 … cl - 1`` with
``cl = floor(2 / p)``, which visits a fraction ``2 / cl ≥ p`` of the entries. When ``2 / p``
is not an integer, a candidate is then kept only if a hash of its position is below a
threshold, which brings the density down to exactly ``p``. The hash depends only on the
seed and the position, so every product, on every backend, regenerates the same
connections.
