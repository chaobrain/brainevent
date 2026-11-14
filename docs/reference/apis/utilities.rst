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

``CSRConfig`` holds the tuning parameters of the hybrid CSR scheduler (an
``MVSchedule``) and, optionally, the benchmark records that chose them. It
describes no particular sparse structure, so one configuration fits every
matrix, and it adds no leaves to a matrix's pytree.

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
plus the L2 cache size of the target GPU. The kernel derives its long-row
mask from ``indptr`` on every call. A matrix built eagerly, on the CPU or
inside ``jax.jit``, and a ``vmap`` of MV, therefore all run the same native
kernel, and structures of equal shape share one compiled program.

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
   MVSchedule

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
Event activity remains a separate setting, defaulting to 0.4%.

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
