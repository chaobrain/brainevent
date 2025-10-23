``brainevent`` documentation
==================================

`BrainEvent <https://github.com/chaobrain/brainevent>`_ provides a set of data structures and algorithms for event-driven computation, which can be used to
model the brain dynamics in a more efficient and biologically plausible way.

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainevent[cpu]

    .. tab-item:: GPU (CUDA)

       .. code-block:: bash

          pip install -U brainevent[cuda12]

          pip install -U brainevent[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainevent[tpu]

----


What is BrainEvent?
^^^^^^^^^^^^^^^^^^^

The brain is fundamentally an **event-driven system**, where discrete spiking events are the primary units of computation.
Traditional dense matrix operations process all array elements, even zeros, leading to significant computational waste
in sparse spike-based scenarios where only a small fraction of neurons are active at any given time.

**BrainEvent** addresses this challenge by:

- **Processing only active events**: Computations skip zero elements, focusing only on neurons that fire spikes
- **Hardware acceleration**: Optimized custom kernels for CPU, GPU, and TPU
- **Seamless JAX integration**: Full support for automatic differentiation, JIT compilation, and vmap
- **Biologically plausible**: Mirrors the sparse, event-driven nature of real neural systems

Why BrainEvent?
^^^^^^^^^^^^^^^

**Performance**: Event-driven computation can be orders of magnitude faster than dense operations when spike activity is sparse (typical in biological neural networks).

**Memory efficiency**: Sparse data structures reduce memory footprint for large-scale neural simulations.

**Biological realism**: Event-based processing naturally represents how the brain computes with discrete action potentials.

**Scalability**: Optimized implementations across different hardware backends (CPU/GPU/TPU) enable simulations from small networks to brain-scale models.


Core Components
^^^^^^^^^^^^^^^

**1. Event Representation**
  BrainEvent provides specialized array types for representing neural events:

  - ``EventArray``: Binary arrays representing spike events (1 = spike, 0 = no spike)
  - ``BinaryArray``: General binary data with event-aware operations
  - ``MaskedFloat``: Float arrays with masking for sparse computations

**2. Sparse Data Structures**
  Multiple sparse matrix formats optimized for event-driven computation:

  - ``COO`` (Coordinate format): Flexible format for constructing sparse matrices
  - ``CSR`` / ``CSC`` (Compressed Sparse Row/Column): Fast row/column-oriented operations
  - ``BlockCSR`` / ``BlockELL``: Block-structured sparse formats for regular connectivity

**3. Just-In-Time Connectivity**
  Generate connectivity matrices on-the-fly without storing full weight matrices (memory-efficient for large networks):

  - ``JITCHomoR`` / ``JITCHomoC``: Homogeneous (constant) weights
  - ``JITCNormalR`` / ``JITCNormalC``: Normally distributed weights
  - ``JITCUniformR`` / ``JITCUniformC``: Uniformly distributed weights

**4. Fixed Connectivity Patterns**
  Specialized structures for biologically realistic fixed-degree connectivity:

  - ``FixedPostNumConn``: Fixed number of post-synaptic connections per pre-synaptic neuron
  - ``FixedPreNumConn``: Fixed number of pre-synaptic connections per post-synaptic neuron

**5. Custom Kernel Framework**
  Extensible system for defining high-performance custom operators:

  - **Numba**: CPU-optimized operations with ``@numba_kernel`` decorator
  - **Warp**: NVIDIA GPU operations using Warp language
  - **Pallas**: TPU/GPU operations using JAX Pallas
  - **XLA Integration**: ``XLACustomKernel`` for custom XLA operators

**6. Synaptic Plasticity**
  Built-in support for learning and plasticity rules:

  - ``csr_on_pre`` / ``csr_on_post``: CSR-based plasticity updates
  - ``coo_on_pre`` / ``coo_on_post``: COO-based plasticity updates
  - ``dense_on_pre`` / ``dense_on_post``: Dense matrix plasticity

**7. Unit-Aware Computation**
  Fully compatible with `BrainUnit <https://github.com/chaobrain/brainunit>`_ for physical unit tracking and dimensional analysis.



Quick Start
^^^^^^^^^^^

**Basic Usage**

To use event-driven computation, wrap your spike arrays with ``EventArray``:

.. code-block:: python

   import brainevent
   import jax.numpy

   # Create spike events (binary array)
   spikes = brainevent.EventArray(jax.numpy.array([1, 0, 1, 0, 1]))

   # Create a sparse connectivity matrix
   conn = brainevent.CSR(...)

   # Event-driven matrix multiplication
   output = spikes @ conn

BrainEvent automatically optimizes computations when ``EventArray`` is involved,
processing only the active (non-zero) events.

**Working with Different Data Structures**

.. code-block:: python

   import brainevent
   import jax.numpy

   # Sparse matrices
   csr_matrix = brainevent.CSR(...)
   coo_matrix = brainevent.COO(...)

   # Just-in-time connectivity (memory efficient)
   jitc_conn = brainevent.JITCHomoR(num_pre=1000, num_post=1000,
                             prob=0.1, weight=0.5, seed=0)

   # Fixed connectivity patterns
   fixed_conn = brainevent.FixedPostNumConn(num_pre=1000, num_post=1000,
                                     conn_num=100, weight=0.5, seed=0)

   # Event-driven computations work with all structures
   spikes = brainevent.EventArray(jax.numpy.array([...]))
   output = spikes @ jitc_conn  # Only active spikes are processed

**Integration with BrainPy**

BrainEvent seamlessly integrates with BrainPy for large-scale brain simulations:

.. code-block:: python


   import brainpy
   import brainunit
   import brainevent

   # Define a spiking neural network
   class Network(brainpy.DynamicalSystem):
       def __init__(self):
           super().__init__()
           self.neurons = brainpy.neurons.LIF(4000, V_rest=-60*brainunit.mV, ...)
           self.synapses = brainevent.CSR(...)  # Use BrainEvent sparse matrix

       def update(self, inp):
           spikes = brainevent.EventArray(self.neurons.spike)
           self.neurons.input += spikes @ self.synapses


See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^


``brainevent`` is one part of our `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.


----


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   Tutorials/01_eventarray_basics.ipynb
   Tutorials/02_sparse_matrices.ipynb
   Tutorials/03_jit_connectivity.ipynb
   Tutorials/04_fixed_connections.ipynb
   Tutorials/05_synaptic_plasticity.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   apis/eventarray.rst
   apis/datastructure.rst
   apis/operator.rst

