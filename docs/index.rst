``brainevent`` documentation
============================

`brainevent <https://github.com/chaobrain/brainevent>`_ provides a set of data structures and algorithms for event-driven computation, which can be used to
model the brain dynamics in a more efficient and biologically plausible way.





----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainevent[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainevent[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainevent[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

----


See also the brain modeling ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `Brain Modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   apis/brianevent.rst

