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


Configuration and mirror preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CSR(..., config=None)`` selects saved scheduler tuning once, during
construction, from the device kind holding the structure and the matrix's
stored shape and edge count. Selection uses an explicit configuration or
environment override first, followed by a matching local workload, another
valid local configuration for the same GPU, and finally library defaults.
Passing ``config=CSRConfig()`` explicitly uses scheduler defaults. Matrix
operations pass the selected configuration to their raw CSR operators
automatically.

Matching uses static metadata and reads no device arrays, so a matrix built
from traced arrays inside ``jax.jit`` selects its tuning the same way. The
configuration describes no structure: the CUDA hybrid MM is planned from
static shapes and the target GPU when it is lowered, and builds its long-row
mask on every call. Raw calls such as ``binary_csrmm(..., config=None)``
run the same native kernel; their ``config`` only selects tuning. See
:doc:`utilities`.

The mirror (the other compression of the structure) is built on first
concrete use and shared by same-structure objects. A structure built inside
``jax.jit`` from traced arrays has none, so ``W @ events`` gathers instead of
scattering; this warns once only when ``cuda_raw`` was chosen explicitly or
globally. Use
``build_weight_indices()`` before passing a matrix as a JIT argument when its
operations need the mirror. A CSR or CSC constructor's
``precompute_weight_indices=True`` option also builds it eagerly:

.. code-block:: python

   >>> import jax.numpy as jnp
   >>> from brainevent import CSR
   >>> matrix = CSR.fromdense(jnp.array([[1., 0.], [2., 3.]]))
   >>> 'csc' in matrix.buffers
   False
   >>> matrix = matrix.build_weight_indices()
   >>> 'csc' in matrix.buffers
   True


Event weight gradients
~~~~~~~~~~~~~~~~~~~~~~

The weight gradient of an event product is the exact derivative of its
forward, so it is decided by the event type alone. A binary event
(:class:`~brainevent.BinaryArray`, a boolean array, or ``weighted=False``)
contributes its activity mask: one for any nonzero event and zero otherwise.
For example, events ``[0., 2., -3.]`` contribute gradient factors
``[0., 1., 1.]``, matching the binary forward. A
:class:`~brainevent.WeightedArray` (or ``weighted=True``) multiplies the
weights by the event values in the forward and contributes those values,
``[0., 2., -3.]``, to the weight gradient. Wrap events in ``WeightedArray``
when value-weighted gradients are wanted.

Binary event gradients enter SDDMM with ``mode='binary'``, weighted events
with ``mode='weighted'``, and ordinary floating-point CSR products with
``mode='float'``. Inactive events contribute exactly zero, even for NaN or
infinite cotangents. The former ``backward_algorithm`` option (TCSR's
``'bptt'``/``'pp_prop'``) was removed; passing it raises a ``TypeError``.


TCSR construction and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`TCSR` inherits :class:`CSR`. Its ordinary constructor stably sorts
columns within each row and reorders heterogeneous weights with their edges.
Duplicate coordinates remain separate edges; scalar and singleton weights
stay compact. The source CSR is unchanged, and the applied row-local
permutation is available as ``matrix.sort_permutation``. Construction builds no
tile geometry. Tiles for a batch width are built lazily, on the first eager
product with that width, and only when the backend can reach CUDA:

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

Native tiled TCSR MM uses its own prepared tile geometry. Calls that select
the inherited CSR hybrid MM route need no preparation.

Geometry is built per tile width on first concrete use and shared by
same-structure objects; the mirror direction is built only after CSR supplies
its mirror. Inside ``jax.jit`` with the matrix as an argument, the structure is
traced and cannot be tiled, so a width that was not prepared uses the parent
CSR path with a one-time :class:`~brainevent.PerformanceWarning`. Call
``matrix.prepare_binary_mm(batch, transpose=..., dtype=...)`` (or run one
eager product) before ``jax.jit`` to keep the tiled kernel. Tiles need a GPU
of compute capability 6.0 or newer.

CUDA selection respects the actual compilation platform and public backend
preferences. Cached resource checks select the parent route before native
execution when the tile does not fit. A lowering without a definite device
assignment conservatively uses the capability intersection of its local
backend candidates.

Factory argument names match the public input type: ``csr`` for ``TCSR``,
``fromcsr`` and ``from_sorted_csr``; ``csc`` for ``fromcsc``; ``mat`` for
``fromdense``. ``fromdense`` retains CSR's ``nse``, ``index_dtype``,
``indptr_dtype``, ``backend``,
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
same-structure geometry do not read host structure.

Inherited methods retain their CSR argument names: ``slice_rows(index)``,
``with_data(data)``, ``apply(fn)``, ``dt2t(y_dim_arr, w_dim_arr)`` and
``dt2t_transposed(y_dim_arr, w_dim_arr)``. The two dt2t methods multiply each
edge value by its row or column value, respectively, and return values in the
stored edge order. ``update_on_*``, ``apply_on_*``, ``delay_layout`` and
``solve`` retain CSR's input restrictions and backend support. 

External per-edge arrays such as ``edge_state``, ``delays`` and ``w_dim_arr``
are not reordered by TCSR construction. Align them with the sorted stored
edge order before passing them to inherited methods:
``edge_state[matrix.sort_permutation]`` (when ``sort_permutation`` is not
``None``) follows the stored edges, since
``matrix.indices == source.indices[matrix.sort_permutation]``. To keep a
custom order instead, apply your own row-local stable permutation to the
source's heterogeneous weights and every external per-edge array, then use
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
