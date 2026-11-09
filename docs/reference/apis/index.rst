Python API
==========

Information-oriented, exhaustive reference for the public ``brainevent`` API. For learning
material, see the :doc:`Data tutorials </tutorials/data-structures/index>` and
:doc:`Events tutorials </tutorials/events/index>`; for task recipes, see the
:doc:`how-to guides </how-to/data-structures/index>`.

Use the categories below to locate the symbols involved in your computation. Event array
types describe activity, while sparse matrix data structures describe connectivity and
its stored or generated representation. Matrix operations list the functions that
combine these inputs, and the custom kernel framework documents the interfaces used to
extend execution with your own implementations. Utility functions support tasks such as
index conversion, configuration controls runtime choices, and error classes help
identify failures. Start with the category closest to the object or operation you are
using, then follow individual symbol pages for signatures and parameter documentation.
When comparing related functions, check their expected shapes and supported
representations before substituting one for another. Pair this reference with a tutorial
or how-to guide when you need a complete example that combines several API components.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Event array types
      :link: events
      :link-type: doc

      Look up representations of binary and other event activity.

   .. grid-item-card:: Sparse matrix data structures
      :link: sparsedata
      :link-type: doc

      Find stored and generated connectivity representations.

   .. grid-item-card:: Matrix operations
      :link: operations
      :link-type: doc

      Browse sparse and event-driven matrix operations.

   .. grid-item-card:: Custom kernel framework
      :link: operator
      :link-type: doc

      Find kernel registration and backend extension interfaces.

   .. grid-item-card:: Error classes
      :link: errors
      :link-type: doc

      Inspect the exception hierarchy and specific failure types.

   .. grid-item-card:: Utility functions
      :link: utilities
      :link-type: doc

      Locate index conversions and supporting functions.

   .. grid-item-card:: Configuration
      :link: config
      :link-type: doc

      Look up runtime controls and configuration functions.

.. toctree::
   :maxdepth: 1
   :caption: Python API

   events
   sparsedata
   operations
   operator
   errors
   utilities
   config
