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

``CSRConfig`` combines a snapshot of persisted hybrid scheduler parameters
with prepared primary/mirror MM structure plans. CSR objects load the tuning
snapshot during construction; operator calls derive their launch parameters
from that object without reading persistent configuration.

``CSR(..., config=None)`` loads persisted tuning once and prepares the
concrete stored traversal. Pass ``CSR(..., config=CSRConfig(...))`` to use
explicit tuning parameters; the constructor still prepares that structure.
For CUDA hybrid MM, construct the matrix outside ``jax.jit`` with concrete
structure on the target GPU, or pass the preparation ``device`` explicitly.
Saving another configuration later does not change existing matrices or their
compiled scheduling snapshots. MM structure statistics and masks are prepared
from the concrete primary and mirror layouts; batch size, value dtype and
weighted scratch sizes are resolved when the operator's call signature is
known, including the effective batch size produced by ``vmap``.

``M.with_rebuilt_config(device=...)`` rebuilds structure and device plans using
``M.config``'s existing tuning parameters, then returns a new matrix holding
the rebuilt configuration. It shares the original values, sparse structure
and structure caches, without reloading persisted tuning or mutating ``M``.
The return value is a matrix, rather than a ``CSRConfig`` object. This method
is available on CSR, CSC and FixedNum matrices.

Use it outside ``jax.jit`` with concrete structure, for example to restore
plans after a traced structural change or prepare configuration for a target
GPU. An explicit ``device`` selects where configuration is prepared and its
plan arrays are placed; it does not migrate matrix values or indices.
Preparation is also preserved through structure-preserving operations such
as ``with_data``; new sparse structures need new plans.

Raw functions such as ``binary_csrmm`` receive arrays, not a CSR object.
They do not look up an owning matrix's configuration, even when the arrays
come from that matrix. Pass its prepared ``M.config`` explicitly.

.. important::

   Before a nonempty CUDA hybrid scatter MM call to ``binary_csrmm`` or
   ``binary_csrmm_indexed`` (``transpose=True`` with a backend resolving to
   ``cuda_raw``), construct a configuration containing the matching structure
   plan, long-row mask and target GPU information, outside ``jax.jit``.
   Pass that configuration as ``config=...`` on the MM call.

   Omitting ``config``, passing ``config=None``, or passing only
   ``config=CSRConfig()`` does not satisfy this requirement. These raw calls
   raise ``KernelCompilationError`` when constructing the hybrid MM kernel.
   This also applies to weighted events and to ``backend=None`` when it
   selects ``cuda_raw``.

Constructing a CSR object is one way to prepare the configuration before
calling raw MM. For the scatter operation ``M.T @ events``, the events have shape
``(M.shape[0], batch)`` and the result has shape ``(M.shape[1], batch)``:

.. code-block:: python

   >>> import jax
   >>> import jax.numpy as jnp
   >>> import brainevent
   >>> device = jax.devices("gpu")[0]
   >>> with jax.default_device(device):
   ...     M = brainevent.CSR(
   ...         jnp.array([1., 2., 3.], dtype=jnp.float32),
   ...         jnp.array([0, 2, 1], dtype=jnp.int32),
   ...         jnp.array([0, 2, 3], dtype=jnp.int32),
   ...         shape=(2, 3), backend="cuda_raw",
   ...         config=brainevent.CSRConfig(), device=device)
   ...     events = jnp.ones((2, 4), dtype=jnp.bool_)
   >>> result = brainevent.binary_csrmm(
   ...     M.data, M.indices, M.indptr, events, shape=M.shape,
   ...     transpose=True, backend="cuda_raw", config=M.config)
   >>> result.shape
   (3, 4)

For these raw binary CSR operators, the default ``config=None`` supplies
fixed MV scheduler constants only. It does not load saved tuning, query GPU
properties, scan ``indptr``, or construct a long-row mask or MM structure
plan. A bare ``CSRConfig()`` has the same unprepared state. Calling
``CSRConfig.prepare_structure(...)`` builds the mask from concrete structure,
but does not query a GPU; its device metadata must also be supplied before
using the resulting configuration for CUDA hybrid MM.

This preparation requirement applies to nonempty CUDA hybrid scatter MM.
Direct gather MM (``transpose=False``), JAX/Numba/cuSPARSE paths, and empty
operations do not require that plan. Ordinary MV can use ``config=None``;
if ``vmap`` converts MV into nonempty CUDA hybrid MM, prepare and pass the
configuration before executing the transformed call. A batch size of one
still uses MM when the raw input is rank two.

A prepared snapshot belongs to its compressed traversal. Raw callers must
pass it with the matching sparse arrays: equal shape and ``nnz`` alone do not
prove that row statistics or the long-row mask match. Changing values is
compatible; replacing the structure requires preparing a new snapshot.

Scratch is allocated by the operator for each call; the four binary CSR
functions no longer accept ``workspace``. MM derives its capacity from
``config`` and the effective batch, dtype and event mode, so its ordinary and
indexed entry points no longer accept ``max_row_nnz``. MV retains this optional
trusted row-length bound for raw calls without prepared statistics.

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
