Utility Functions
=================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


Index Conversion
----------------

.. autosummary::
   :toctree: generated/

   csr_to_coo_index
   csr_to_csc_index
   csc_to_csr_index
   coo_to_csc_index
   coo2csr


GPU/TPU Random Number Generator
-------------------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   PallasLFSR88RNG
   PallasLFSR113RNG
   PallasLFSR128RNG

.. autosummary::
   :toctree: generated/

   PallasLFSRRNG
   get_pallas_lfsr_rng_class


.. _csr-mm-configuration:

CSR Configuration and Planning
------------------------------

``CSRConfig`` holds the four MV scheduler constants and five MM chunk
policies directly. It describes no particular sparse structure, so one
configuration fits every matrix, and it adds no leaves to a matrix's pytree.
All nine scheduling fields participate in equality, hashing and JIT identity;
benchmark records are dictionaries excluded from that identity.

The existing positional fields remain ``block_size=512``,
``fixed_scatter_blocks=1024``, ``tpr_threshold=128`` and ``task_nnz=512``.
The MM policies are keyword-only:

.. list-table:: MM chunk policies
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Default
     - Meaning
   * - ``mm_l2_divisor``
     - ``4``
     - L2 divisor for the compact output chunk.
   * - ``mm_sparse_l2_divisor``
     - ``2``
     - L2 divisor for four-lane schedules on large caches.
   * - ``mm_sparse_l2_min_bytes``
     - ``32 << 20``
     - Minimum L2 size for the sparse allowance, in bytes.
   * - ``mm_short_rows_floor``
     - ``1 << 20``
     - Minimum short-list allowance, in int32 entries.
   * - ``mm_max_chunk``
     - ``65535``
     - Maximum batches per chunk, in ``[1, 65535]``.

``config.to_dict()`` serializes these nine fields; ``CSRConfig.from_dict``
restores them and defaults missing fields, including MM policies omitted by
older four-field configuration files. Resolved launch plans and structure
masks are not serialized with the configuration.

``CSR(..., config=None)`` loads the persisted tuning for the matrix's shape,
stored-edge count and device kind once, during construction. An explicit
``CSR(..., config=CSRConfig(...))`` uses the given parameters instead.
Selection reads no array values, so a matrix built from traced arrays inside
``jax.jit`` selects its tuning the same way. Saving another configuration
later does not change existing matrices.

The native CUDA hybrid scatter MM of ``binary_csrmm`` and
``binary_csrmm_indexed`` (``transpose=True`` with a backend resolving to
``cuda_raw``) is planned when the call is lowered. The plan uses static
shapes only (rows, columns, stored edges, batch, value dtype and event mode)
plus the L2 cache size of the target GPU. Matrix objects provide their cached
long-row mask as a dynamic operand; raw calls that omit ``long_mask`` derive
it from ``indptr`` automatically. A matrix built eagerly, on the CPU or
inside ``jax.jit``, and a ``vmap`` of MV, therefore all run the same native
kernel. Structures with equal shapes and the same scheduling policies share
one compiled program.

The output batch chunk uses a quarter of L2, or half for the four-lane
schedule on devices with at least 32 MiB L2. Short-row lists share that byte
budget, with a minimum allowance of 4 MiB or one int32 entry per stored edge,
whichever is larger; at least one batch always fits. This avoids tiny launch
groups on large sparse matrices while bounding short-list scratch for tall, narrow
outputs. The long-row classification and the K1/K2/K3 execution remain the
same. A larger chunk is chosen only when it removes launch groups;
otherwise the smaller, fixed-allowance chunk is retained. These are the
default policies; customizing a divisor, allowance or chunk cap changes the
chunk calculation without changing the lane group or long-row threshold.
The MM geometry remains fixed at 16 iterations and 512 edges per long task.

``config.mm_plan(rows, cols, nnz, batch, dtype, weighted, l2_bytes)`` returns
a fresh dictionary with ``group``, ``long_threshold``, ``long_tasks``,
``chunk``, ``task_capacity``, ``workspace_words`` and ``short_rows_size``.
It derives these values from the call metadata at lowering and does not
store them on the config. Modifying a returned dictionary cannot change a
later plan. ``CSRConfig.mm_group(rows, nnz)`` and
``CSRConfig.mm_threshold(rows, nnz)`` expose the shared row classification
used by both mask construction and lowering.

Raw functions such as ``binary_csrmm`` receive arrays, not a CSR object.
Their ``config=None`` uses the default scheduler constants and never loads
saved tuning; pass ``M.config`` to use a matrix's tuning. For the scatter
operation ``M.T @ events``, the events have shape ``(M.shape[0], batch)`` and
the result has shape ``(M.shape[1], batch)``:

.. code-block:: python

   >>> import jax.numpy as jnp
   >>> import brainevent
   >>> M = brainevent.CSR(
   ...     jnp.array([1., 2., 3.], dtype=jnp.float32),
   ...     jnp.array([0, 2, 1], dtype=jnp.int32),
   ...     jnp.array([0, 2, 3], dtype=jnp.int32),
   ...     shape=(2, 3))
   >>> events = jnp.ones((2, 4), dtype=jnp.bool_)
   >>> result = brainevent.binary_csrmm(
   ...     M.data, M.indices, M.indptr, events, shape=M.shape,
   ...     transpose=True, config=M.config)
   >>> result.shape
   (3, 4)

Scratch is allocated by the operator for each call; the four binary CSR
functions do not accept ``workspace``. MM sizes its task queue from the
static shapes, so its ordinary and indexed entry points do not accept
``max_row_nnz``. MV keeps this optional trusted bound on the longest row,
which shrinks its task queue.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   CSRConfig

.. autosummary::
   :toctree: generated/

   load_csr_config
   save_csr_config
   init_csr_config


CSR hybrid scheduler tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``init_csr_config()`` explicitly benchmarks the production homogeneous,
float32 binary SpMV kernel. Its default search is the Cartesian product of
``block_size=(512, 256)``, ``fixed_scatter_blocks=(512, 1024, 2048)``,
``tpr_threshold=(128, 256)`` and ``task_nnz=(512, 1024, 2048, 4096)``:
48 candidates for each of matrix densities 3% and 0.3%, at the same shape.
Event activity remains a separate setting, defaulting to 0.4%. Initialization
continues to tune MV candidates; it does not search the MM policies.

Initialization returns a list of workload results. Each result includes
``metadata``, ``best_config``, ``records`` and ``status``; an unsuccessful
workload has no winner. Successful workloads are stored independently in
``csr_hybrid_config.json`` under the target GPU model. Repeating one workload
updates its record while retaining other workloads, GPU models and any
legacy flat fallback. Timing includes host submission and the final device
synchronization; numerical checks run outside the timed interval.

``load_csr_config(matrix_metadata={'shape': (rows, cols), 'nnz': nnz})``
selects an exact stored workload using integer metadata. Without a match,
it tries the same GPU's legacy fallback, then another valid workload in
stable shape/density order, and finally library defaults. An explicit
``config`` passed to a matrix constructor takes precedence over the
``BRAINEVENT_CSR_HYBRID_CONFIG`` environment override and persisted tuning.
Existing matrices keep their own snapshots after the saved results change.

``save_csr_config(config, matrix_metadata=metadata)`` saves one workload;
omitting metadata retains the legacy device-fallback interface. The file
uses the current compilation cache directory, which includes the library
version by default; older version directories are not migrated automatically.
Initialization is never triggered by loading or matrix construction.


Benchmarking
------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   BenchmarkConfig
   BenchmarkRecord
   BenchmarkResult

.. autosummary::
   :toctree: generated/

   benchmark_function


Kernel Helpers
--------------

.. autosummary::
   :toctree: generated/

   defjvp
   general_batching_rule
   jaxtype_to_warptype
   jaxinfo_to_warpinfo
