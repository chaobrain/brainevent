``brainevent`` documentation
=============================

`BrainEvent <https://github.com/chaobrain/brainevent>`_ provides data structures and
algorithms for **event-driven computation** on CPUs, GPUs, and TPUs. By processing only
the active (non-zero) spikes in a network, it models brain dynamics far more efficiently
than dense matrix operations — while integrating seamlessly with JAX's autodiff, JIT, and
``vmap``.

.. code-block:: python

   import brainevent
   import jax.numpy as jnp

   spikes = brainevent.BinaryArray(jnp.array([1, 0, 1, 0, 1]))
   conn = brainevent.JITCScalarR(num_pre=5, num_post=3, prob=0.5, weight=0.2, seed=0)

   output = spikes @ conn        # only active spikes are processed

----

Where to go next
^^^^^^^^^^^^^^^^^

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: getting-started/installation
      :link-type: doc

      Install ``brainevent`` and run your first event-driven computation in 60 seconds.

   .. grid-item-card:: 📘 Tutorials
      :link: tutorials/data-structures/index
      :link-type: doc

      Learning-oriented, step-by-step notebooks — from event arrays to writing your own
      custom kernels.

   .. grid-item-card:: 🛠️ How-to Guides
      :link: how-to/data-structures/index
      :link-type: doc

      Task-oriented recipes for concrete problems: choosing a sparse format, building a
      network, compiling raw CUDA.

   .. grid-item-card:: 💡 Explanation
      :link: explanation/event-driven-computation
      :link-type: doc

      Understanding-oriented background: the event-driven model, sparse-format trade-offs,
      and the FAQ.

   .. grid-item-card:: 📖 Reference
      :link: reference/apis/index
      :link-type: doc

      Information-oriented API and kernel reference, plus the changelog.

   .. grid-item-card:: 🌐 Ecosystem
      :link: https://brainx.chaobrain.com/

      ``brainevent`` is one part of the `BrainX <https://brainx.chaobrain.com/>`_ brain
      modeling ecosystem.

----

.. toctree::
   :hidden:
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/data-structures/index
   tutorials/custom-operators/index

.. toctree::
   :hidden:
   :caption: How-to Guides

   how-to/data-structures/index
   how-to/building-extending/index

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/event-driven-computation
   explanation/sparse-formats
   explanation/jit-connectivity
   explanation/custom-kernel-architecture
   explanation/faq

.. toctree::
   :hidden:
   :caption: Reference

   reference/apis/index
   reference/kernels/index
   reference/changelog
