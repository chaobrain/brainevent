Caching
=======

``brainevent.kernix`` caches compiled shared libraries to avoid redundant
recompilation.  The cache key is a SHA-256 hash of:

- Source code
- Compiler flags (including ``optimization_level`` and ``use_fast_math``)
- GPU architecture (or ``"cpu"`` for C++ builds)
- Compiler version
- brainevent version

Cache Directory
---------------

By default, cached artefacts are stored under ``~/.cache/brainevent/``.
You can change this in three ways:

1. **Python API** (recommended):

   .. code-block:: python

      from brainevent import kernix

      kernix.set_cache_dir("/tmp/my_brainevent_cache")
      print(kernix.get_cache_dir())  # /tmp/my_brainevent_cache

2. **Environment variable**:

   .. code-block:: bash

      export BRAINEVENT_CACHE_DIR=/tmp/my_brainevent_cache

3. **Per-call override** via ``build_directory``:

   .. code-block:: python

      mod = kernix.load_cuda_inline(
          ...,
          build_directory="/tmp/specific_build",
      )

Clearing the Cache
------------------

.. code-block:: python

   # Clear everything
   kernix.clear_cache()

   # Clear only entries for a specific module
   kernix.clear_cache("my_kernels")

Force Rebuild
-------------

To skip the cache and recompile from scratch:

.. code-block:: python

   mod = kernix.load_cuda_inline(
       ...,
       force_rebuild=True,
   )

Idempotent Re-import
--------------------

``kernix`` tracks which compiled ``.so`` files have already had their FFI
targets registered in the current process.  Calling ``load_cuda_file`` (or
``load_cuda_inline``) a second time with the same source produces a cache
hit — the ``.so`` is loaded but registration is skipped silently.  This
makes it safe to call the load function at module import time without
worrying about ``RegistrationError`` when the same module is imported from
multiple files.
