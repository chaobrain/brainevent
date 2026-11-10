Sparse Matrix Data Structures
=============================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


Data Representation
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DataRepresentation
   Dense


CSR / CSC / TCSR
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CSR
   CSC
   TCSR


TCSR construction and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`TCSR` inherits :class:`CSR`. Its ordinary constructor stably sorts
columns within each row and reorders heterogeneous weights with their edges.
Duplicate coordinates remain separate edges; scalar and singleton weights
stay compact. The source CSR is unchanged. Construction automatically prepares
primary geometry for every supported batch and weight dtype. Mirror geometry
is prepared lazily after CSR supplies the mirror; no explicit tile preparation
is required:

.. code-block:: python

   >>> import jax.numpy as jnp
   >>> from brainevent import BinaryArray, CSR, TCSR
   >>> source = CSR.fromdense(jnp.array([[1., 0.], [2., 3.]]), backend='jax_raw')
   >>> matrix = TCSR(source)
   >>> (BinaryArray(jnp.ones((4, 2), dtype=bool)) @ matrix).shape
   (4, 2)

Both ``matrix @ x`` and ``x @ matrix`` retain CSR's public input and unit
contracts. Their TCSR overrides select Binary CUDA MM only for eligible
inputs, supported actual-target resources and available matching geometry.
MV, WeightedArray and ordinary numerical inputs use the corresponding parent
operator. Binary event wrappers accepted by CSR need no changes.

Primary geometry is initialized once. The public CSR mirror remains lazy;
when it becomes available, concrete mirror geometry is prepared once and
shared by same-structure objects. Traced structures without mirror geometry
use the parent path without reading host structure. Changing batch does not
require preparing another plan. ``prepare_binary_mm`` remains an optional
legacy compatibility method and does not bind geometry to a batch.

CUDA selection respects the actual compilation platform and public backend
preferences. Cached resource checks select the parent route before native
execution when the tile does not fit. A lowering without a definite device
assignment conservatively uses the capability intersection of its local
backend candidates.

Factory argument names match the public input type: ``csr`` for ``TCSR``,
``fromcsr`` and ``from_sorted_csr``; ``csc`` for ``fromcsc``; ``mat`` for
``fromdense``. ``fromdense`` retains CSR's ``nse``, ``index_dtype``,
``indptr_dtype``, ``backend``, ``backward_algorithm``,
``precompute_weight_indices``, ``config`` and ``device`` parameters.

.. list-table:: TCSR API migration
   :header-rows: 1
   :widths: 32 38 30

   * - Previous API
     - Current API
     - Compatibility
   * - ``TCSR(source=csr)``
     - ``TCSR(csr=csr)``
     - ``source`` remains a keyword alias. Passing both sources raises ``TypeError``.
   * - ``fromcsr(source=csr)``, ``fromcsc(source=csc)``
     - ``TCSR.fromcsr(csr=csr)``, ``TCSR.fromcsc(csc=csc)``
     - Thin conversion factories retain the ``source`` alias and bind configuration to the new sorted structure.
   * - ``from_sorted_csr(source=csr)``
     - ``TCSR.from_sorted_csr(csr=csr)``
     - Sortedness is trusted. Concrete new sources prepare primary geometry; complete same-structure geometry is reused without host reads.
   * - ``fromdense(source=mat)``
     - ``TCSR.fromdense(mat=mat)``
     - The ``source`` alias remains; passing both source arguments raises ``TypeError``.
   * - ``binary_backend='jax'``
     - ``backend='jax_raw'``
     - One backend preference is stored. The legacy alias and spelling are normalized; conflicting explicit values raise ``ValueError``.
   * - ``materialize_tcsc_mirror()``
     - ``matrix = matrix.build_weight_indices()``
     - The legacy method delegates to CSR and returns its result. ``has_tcsc_mirror`` only queries the existing cache.
   * - Private transposed TCSR views
     - ``matrix.T`` is CSC; ``matrix.T.T`` is CSR.
     - Transposition and format conversion follow CSR's type and storage contracts.

``from_sorted_csr`` shares the source ``CSRConfig`` even when its ``backend``
preference is overridden. It skips sorting; a concrete source without primary
geometry prepares it. Internal pytree restoration and reuse of complete
same-structure geometry do not read host structure. This factory does not
implicitly rebuild device configuration. When switching a configuration without GPU
device information to ``cuda_raw``, first place the matrix arrays on the
target GPU, then call ``matrix.with_rebuilt_config(device=target_device)``
outside JIT before using a parent MM path that requires GPU configuration,
including a fallback with no matching tile plan. ``device`` can be omitted
when structure placement already identifies the target GPU. Rebuilding
configuration does not move matrix or tile arrays. Use ``TCSR(csr, backend=...)`` or
``TCSR.fromcsr(csr, backend=...)`` when configuration should be rebound during
construction.

Inherited methods retain their CSR argument names: ``slice_rows(index)``,
``with_data(data)``, ``apply(fn)``, ``dt2t(y_dim_arr, w_dim_arr)`` and
``dt2t_transposed(y_dim_arr, w_dim_arr)``. The two dt2t methods multiply each
edge value by its row or column value, respectively, and return values in the
stored edge order. ``update_on_*``, ``apply_on_*``, ``delay_layout`` and
``solve`` retain CSR's input restrictions and backend support. Existing
gradient options are inherited from a CSR source or accepted by
``fromdense``; this integration adds no pp_prop or BPSA implementation.

External per-edge arrays such as ``edge_state``, ``delays`` and ``w_dim_arr``
are not reordered by TCSR construction. Align them with the sorted stored
edge order before passing them to inherited methods. To retain existing
state, apply the same row-local stable permutation to the source's
heterogeneous weights and every external per-edge array, then use
``from_sorted_csr``. Preserve each duplicate edge separately. Old mirror-slot
state also needs alignment with the new CSC view's canonical storage.

The class overrides construction, ``fromdense``, ``slice_rows``, ``diag_add``,
both public matmul operators, ``_mirror``, ``tree_flatten`` and
``tree_unflatten``. The mirror hook invokes CSR first and only adds lazy tile
preparation. ``_stored_event_product`` is inherited directly, so a public
operator fallback cannot re-enter TCSR MM dispatch. Compatibility factories
and mirror aliases reuse these paths without introducing separate storage,
mirror or differentiation algorithms.

``tocoo()`` preserves every duplicate coordinate and its value, just as CSR
does. In the current CUDA validation environment, the downstream JAX
``COO.todense()`` operation does not sum duplicate coordinates and may retain
only one value. Coalesce duplicates first or use ``CSR.todense()`` /
``TCSR.todense()`` when duplicate edges must be accumulated. This downstream
restriction is shared with CSR.


JITC Base
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   JITCMatrix
   JITCScalarMatrix


JITC Scalar
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   JITCScalarR
   JITCScalarC


JITC Normal
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   JITCNormalR
   JITCNormalC


JITC Uniform
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   JITCUniformR
   JITCUniformC


Fixed Connectivity
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FixedNumConn
   FixedNumPerPre
   FixedNumPerPost
