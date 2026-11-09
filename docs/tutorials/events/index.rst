Events
======

Learning-oriented, hands-on notebooks for representing discrete neural activity and for
operations whose work is driven by active events. Regular array computations appear only
as correctness or performance baselines.

If you are brand new, start with
:doc:`/getting-started/quickstart`. For connectivity storage and
generation, use the :doc:`Data tutorials </tutorials/data-structures/index>`.

Use these lessons to connect the representation of neural activity with the computations
that consume it. Start with binary events to see how an activity array expresses which
neurons are active and how those events interact with connectivity. Then continue to
synaptic plasticity to explore updates that depend on activity over time. Run the
examples in order within each notebook, inspecting intermediate arrays before moving to
larger computations. Pay attention to the difference between an event representation and
a connectivity representation: the former describes activity, while the latter
determines where that activity is sent. If a connectivity format is unfamiliar, consult
the data tutorials alongside the lesson. Together, these notebooks provide a foundation
for choosing event operations and understanding the inputs that a network or learning
rule needs.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Binary events
      :link: binary-events
      :link-type: doc

      Represent discrete activity and apply event-driven computations.

   .. grid-item-card:: Synaptic plasticity
      :link: synaptic-plasticity
      :link-type: doc

      Explore activity-dependent synaptic updates in a guided notebook.

.. toctree::
   :maxdepth: 1
   :caption: Events

   binary-events
   synaptic-plasticity
