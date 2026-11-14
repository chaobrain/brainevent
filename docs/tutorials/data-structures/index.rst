Data
====

Learning-oriented, hands-on notebooks for structures that represent, store, or generate
neural connectivity. Start with general compressed sparse storage, continue to explicit
fixed-count constraints, and finish with connectivity generated just in time.

If you are brand new, start with :doc:`/getting-started/quickstart`.

For binary activity representations and event-driven behavior, use the
:doc:`Events tutorials </tutorials/events/index>`.

Use this section to learn how a connectivity representation changes the way you
construct and apply a neural network. The notebooks introduce the stored arrays and
their relationships before showing how to use each structure in computations. Begin with
sparse matrices to understand compressed storage and the distinction between row and
column layouts. Continue with fixed connections when your model constrains the number of
connections, then explore just-in-time connectivity when storing every connection would
be inconvenient. Work through the examples with a small network first so you can inspect
shapes and compare results. When choosing a representation for a larger model, consider
both the connectivity pattern and the operations you need to perform. The linked how-to
guides can then help you turn these lessons into a concrete implementation.

Explore this section
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Sparse matrices
      :link: 02_sparse_matrices
      :link-type: doc

      Learn compressed sparse storage and how to use sparse matrix representations.

   .. grid-item-card:: Fixed connections
      :link: 04_fixed_connections
      :link-type: doc

      Explore connectivity structures with explicit connection-count constraints.

   .. grid-item-card:: Just-in-time connectivity
      :link: 03_jit_connectivity
      :link-type: doc

      Generate connectivity during computation instead of storing every connection.

.. toctree::
   :maxdepth: 1
   :caption: Data

   02_sparse_matrices
   04_fixed_connections
   03_jit_connectivity
