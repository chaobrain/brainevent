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


CSR Configuration and Planning
------------------------------

``CSRConfig`` combines a snapshot of persisted hybrid scheduler parameters
with prepared primary/mirror MM structure plans. CSR objects load the tuning
snapshot during construction; operator calls derive their launch parameters
from that object without reading persistent configuration.

``CSR(..., config=None)`` loads persisted tuning once. Pass
``CSR(..., config=CSRConfig(...))`` to use explicit parameters instead.
Saving another configuration later does not change existing matrices or their
compiled scheduling snapshots. MM structure statistics and masks are prepared
from the concrete primary and mirror layouts; batch size, value dtype and
weighted scratch sizes are resolved when the operator's call signature is
known, including the effective batch size produced by ``vmap``.

``M.prepare_config(device=...)`` returns a new matrix sharing the original
values and sparse structure. It prepares plans using ``M.config``'s existing
tuning parameters, without reloading persistence or mutating ``M``. Use it
outside ``jax.jit`` when preparing a matrix for a target GPU. Preparation is
also preserved through structure-preserving operations such as ``with_data``;
new sparse structures need new plans.

Raw functions such as ``binary_csrmm`` accept the same explicit snapshot.
For the scatter operation ``M.T @ events``, the events have shape
``(M.shape[0], batch)`` and the result has shape ``(M.shape[1], batch)``:

.. code-block:: python

   >>> import jax
   >>> import jax.numpy as jnp
   >>> import brainevent
   >>> device = jax.devices("gpu")[0]
   >>> M = brainevent.CSR(
   ...     jnp.array([1., 2., 3.], dtype=jnp.float32),
   ...     jnp.array([0, 2, 1], dtype=jnp.int32),
   ...     jnp.array([0, 2, 3], dtype=jnp.int32),
   ...     shape=(2, 3), backend="cuda_raw",
   ...     config=brainevent.CSRConfig(), device=device)
   >>> events = jnp.ones((2, 4), dtype=jnp.bool_)
   >>> result = brainevent.binary_csrmm(
   ...     M.data, M.indices, M.indptr, events, shape=M.shape,
   ...     transpose=True, backend="cuda_raw", config=M.config)
   >>> result.shape
   (3, 4)

For raw operators, ``config=None`` means fixed default scheduler constants;
it does not load a saved configuration or implicitly scan the structure.
Nonempty hybrid MM execution requires a matching prepared structure and GPU
description, and raises an error if either is absent. Prepare before JIT
tracing. Empty operations and paths that do not select hybrid MM do not
require an MM plan. In particular, ``vmap`` can transform an MV call into MM,
so that transformed call also needs preparation when it selects hybrid MM.

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
