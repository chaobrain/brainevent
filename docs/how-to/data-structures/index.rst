Working with data structures
============================

Task-oriented recipes for picking and using ``brainevent``'s event and sparse data
structures. Each guide assumes you already know the basics from the
:doc:`Data tutorials </tutorials/data-structures/index>` or
:doc:`Events tutorials </tutorials/events/index>` and want to accomplish one concrete
thing.

Start with the recipe that matches the decision or task in front of you. The format
selection guide helps you compare representations before committing to a storage layout.
The large network guide focuses on generating connectivity just in time, which is useful
when explicit connection storage becomes a concern. The plasticity guide addresses
synaptic updates, while the unit-aware computation guide explains how to work with
physical quantities. Read the assumptions and input shapes in each recipe before
adapting its examples to your own model. A representation that suits one connectivity
pattern may require a different approach for another, so consider the operations you
need as well as the size of the network. Return to the tutorials whenever you need a
fuller introduction to the underlying data structures or event representations.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Choose a sparse format
      :link: choosing-a-sparse-format
      :link-type: doc

      Compare connectivity representations for your model and operations.

   .. grid-item-card:: Scale connectivity
      :link: jit-connectivity-large-networks
      :link-type: doc

      Use just-in-time connectivity in larger networks.

   .. grid-item-card:: Update synapses
      :link: synaptic-plasticity
      :link-type: doc

      Apply synaptic plasticity with event and sparse structures.

   .. grid-item-card:: Work with units
      :link: unit-aware-computation
      :link-type: doc

      Use physical quantities in event-driven computations.

.. toctree::
   :maxdepth: 1
   :caption: Working with data structures

   choosing-a-sparse-format
   jit-connectivity-large-networks
   synaptic-plasticity
   unit-aware-computation
