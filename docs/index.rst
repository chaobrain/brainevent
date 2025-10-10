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

    .. tab-item:: GPU (CUDA 13.0)

       .. code-block:: bash

          pip install -U brainevent[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainevent[tpu]

----


See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^


``brainevent`` is one part of our `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   apis/eventarray.rst
   apis/datastructure.rst
   apis/operator.rst

