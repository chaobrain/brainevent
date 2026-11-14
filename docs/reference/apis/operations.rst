Matrix Operations
=================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


CSR Operations
--------------

Binary matrix-vector / matrix-matrix multiplication.

.. note::

   The native CUDA hybrid scatter MM of ``binary_csrmm`` and
   ``binary_csrmm_indexed`` (``transpose=True``, backend resolving to
   ``cuda_raw``) needs no preparation: it is planned from static shapes and
   the target GPU when the call is lowered, including inside ``jax.jit``.
   ``config`` only selects scheduler tuning; see :ref:`csr-mm-configuration`.

.. autosummary::
   :toctree: generated/

   binary_csrmv
   binary_csrmv_p
   binary_csrmm
   binary_csrmm_p

Binary multiplication with a fused weight gather (indexed weights).

.. autosummary::
   :toctree: generated/

   binary_csrmv_indexed
   binary_csrmv_indexed_p
   binary_csrmm_indexed
   binary_csrmm_indexed_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   csrmv
   csrmv_p
   csrmm
   csrmm_p
   csrmv_dt2t
   cscmv_dt2t
   csrmv_dt2t_p
   csrmm_dt2t
   cscmm_dt2t
   csrmm_dt2t_p

Plasticity operations.

.. autosummary::
   :toctree: generated/

   update_csr_on_binary_pre
   update_csr_on_binary_pre_p
   update_csr_on_binary_post
   update_csr_on_binary_post_p
   update_csc_on_binary_pre
   update_csc_on_binary_post
   csr_apply_on_pre
   csr_apply_on_post

Row slicing.

.. autosummary::
   :toctree: generated/

   csr_slice_rows
   csr_slice_rows_p

Per-edge-delay spike delivery into an arrival ring.

Build a :class:`CSRDelayLayout` once, outside the timed loop, with
``CSRDelayLayout.build(...)`` or ``CSR.delay_layout(delays, n_slots)``. It
sorts every row by (delay, column), validates the structure and delays once,
and records where each delay segment starts. ``csr_delay_deliver_layout``
then adds one step's events into an explicit ``(n_slots, n_post)`` ring:
an event of row ``i`` at step ``t`` over an edge of delay ``d`` lands in slot
``(t + d) % n_slots``. Delivery is pure: it returns the updated ring, never
clears a slot, and keeps no state on the layout, so reading and clearing the
current slot (and advancing time) stay with the caller.

Delays lie in ``[0, n_slots)`` and may be one integer for all edges or one
per edge. The layout stores weights in its own edge order; ``with_data`` and
``original_data`` convert from and to the original edge order through
``perm``. The raw constructor accepts prepared arrays, including traced ones
inside ``jax.jit``. The ring keeps its dtype and unit. Binary events add the
weight for every nonzero event; weighted events (``WeightedArray`` or
``weighted=True``) multiply it by the event value. On the GPU the default
backend is a delay-major CUDA kernel, on the CPU an event-driven Numba kernel;
``jax_raw`` is available everywhere. Delivery supports JIT, ``vmap``,
``shard_map``, forward- and reverse-mode differentiation.

.. autosummary::
   :toctree: generated/

   CSRDelayLayout
   csr_delay_deliver_layout
   csr_delay_deliver_layout_p

Dense Operations
----------------

Dense-matrix @ binary-vector / binary-matrix.

.. autosummary::
   :toctree: generated/

   binary_densemv
   binary_densemv_p
   binary_densemm
   binary_densemm_p

Plasticity operations.

.. autosummary::
   :toctree: generated/

   update_dense_on_binary_pre
   update_dense_on_binary_pre_p
   update_dense_on_binary_post
   update_dense_on_binary_post_p


JITC Scalar Operations
----------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_jitsmv
   binary_jitsmv_p
   binary_jitsmm
   binary_jitsmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   jits
   jits_p
   jitsmv
   jitsmv_p
   jitsmm
   jitsmm_p
   jitsmv_dt2t


JITC Normal Operations
----------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_jitnmv
   binary_jitnmv_p
   binary_jitnmm
   binary_jitnmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   jitn
   jitn_p
   jitnmv
   jitnmv_p
   jitnmm
   jitnmm_p
   jitnmv_dt2t


JITC Uniform Operations
------------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_jitumv
   binary_jitumv_p
   binary_jitumm
   binary_jitumm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   jitu
   jitu_p
   jitumv
   jitumv_p
   jitumm
   jitumm_p
   jitumv_dt2t


Fixed Connectivity Operations
-----------------------------

Binary matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   binary_fcnmv
   binary_fcnmv_p
   binary_fcnmm
   binary_fcnmm_p

Float matrix-vector / matrix-matrix multiplication.

.. autosummary::
   :toctree: generated/

   fcnmv
   fcnmm
   fcnmv_dt2t
   fcnmm_dt2t

Plasticity operations.

.. autosummary::
   :toctree: generated/

   update_fixed_pre_conn_on_binary_post
   update_fixed_post_conn_on_binary_pre
   fcn_plasticity_row_p
