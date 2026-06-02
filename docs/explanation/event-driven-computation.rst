What is event-driven computation?
=================================

The brain is fundamentally an **event-driven system**: discrete spiking events are the
primary unit of computation. Traditional dense matrix operations process *every* element —
including the zeros — which is enormously wasteful when only a small fraction of neurons
fire at any given moment.

``brainevent`` addresses this by processing **only active events**:

- **Skip the zeros** — computation visits only the neurons that actually spiked.
- **Hardware acceleration** — optimized custom kernels for CPU, GPU, and TPU.
- **Seamless JAX integration** — full support for automatic differentiation, JIT
  compilation, and ``vmap``.
- **Biologically plausible** — mirrors the sparse, event-driven nature of real neural
  systems.

The central object is :class:`~brainevent.BinaryArray`, which marks an array as a vector or
matrix of spike events (``1`` = spike, ``0`` = no spike). Any matrix multiplication involving
a ``BinaryArray`` is automatically rewritten to an event-driven kernel.


Core components
---------------

**1. Event representation**
  :class:`~brainevent.BinaryArray` represents binary spike events. Indexed variants
  (:class:`~brainevent.IndexedBinary1d`, :class:`~brainevent.IndexedBinary2d`) store only the
  indices of active events.

**2. Sparse data structures**
  Sparse matrix formats optimized for event-driven products:
  :class:`~brainevent.CSR` / :class:`~brainevent.CSC` (compressed sparse row/column). Coordinate
  triplets can be converted to CSR with the :func:`~brainevent.coo2csr` helper. See
  :doc:`sparse-formats`.

**3. Just-in-time connectivity**
  Generate connectivity on the fly without storing the full weight matrix — memory-efficient
  for very large networks. Scalar (:class:`~brainevent.JITCScalarR` / ``…C``), normal
  (``JITCNormalR`` / ``…C``), and uniform (``JITCUniformR`` / ``…C``) weight variants. See
  :doc:`jit-connectivity`.

**4. Fixed connectivity patterns**
  Biologically realistic fixed-degree connectivity:
  :class:`~brainevent.FixedPreNumConn` (fixed number of pre-synaptic connections) and
  :class:`~brainevent.FixedPostNumConn` (fixed number of post-synaptic connections).

**5. Custom kernel framework**
  An extensible system for high-performance custom operators — Numba (CPU), Numba-CUDA and
  Warp (GPU), and raw C++/CUDA via XLA FFI. See :doc:`custom-kernel-architecture`.

**6. Synaptic plasticity**
  Event-driven learning rules for both CSR and dense weights, driven by pre- or
  post-synaptic spikes (``update_csr_on_binary_pre`` / ``…_post`` and the dense
  equivalents).

**7. Unit-aware computation**
  Full compatibility with `BrainUnit <https://github.com/chaobrain/brainunit>`_ for physical
  unit tracking and dimensional analysis.


How the optimization works
---------------------------

When you write ``spikes @ conn`` and ``spikes`` is a ``BinaryArray``, ``brainevent`` does not
form the usual dense product. Instead it dispatches a kernel that iterates over the **active
spike indices** and accumulates only their contributions to the output. The work therefore
scales with the number of spikes, not with the size of the matrix — exactly the asymptotic
advantage that makes large, sparsely-active networks tractable.

This dispatch is transparent: the same ``@`` operator works for dense arrays, ``CSR``/``CSC``,
JITC, and fixed-connectivity structures, and it composes with ``jax.jit``, ``jax.grad``, and
``jax.vmap`` like any other JAX operation.
