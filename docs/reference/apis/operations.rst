Matrix Operations
=================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


CSR Operations
--------------

Binary matrix-vector / matrix-matrix multiplication.

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

Prepared per-edge-delay spike delivery into an arrival ring.

Build a ``delaycsr`` once, either directly or with ``CSR.delay_layout``.
``csr_delay_deliver_layout`` adds to an explicit ring without clearing slots
or mutating the object's owned ring, head, workspace or canonical weights.
The layout inherits the parent CSR backend; explicit backends are
``cuda_raw`` and ``jax_raw``.

For an externally sorted matrix and existing metadata, use
``delaycsr.from_prepared(...)`` instead of the ordinary constructor. Unlike
``sorted_by_delay=True``, this entry skips both sorting and metadata
preparation. Supply the CSR arrays (or a source CSR) and these layout fields:

* ``shape=(M, N)``, unless supplied by a source CSR.
* ``delay_steps``: a static, increasing tuple of nonnegative delays of length
   ``K >= 1``. A JAX array is not a static delay tuple.
* ``delay_count``: shape ``(M, K)``, with the ``indptr`` dtype. Counts are
   nonnegative and sum to each CSR row's edge count.
* ``delay_off``: int32/int64 array of shape ``(K,)`` containing element offsets
   ``delay_steps[k] * N``, not byte offsets. Use int64 if ``L * N`` exceeds
   INT32_MAX.
* ``row_task_count``: int32 array of shape ``(M,)``. Rows with at most 128
   edges require zero tasks. For longer rows, round each delay segment's edge
   count up in chunks of 4096 and sum the resulting task counts.
* ``capacity``: a static Python integer covering the sum of all row task
   counts, including the case where every row fires. Zero is valid.

Edges must be consecutive in ``delay_steps`` order within each row, with
ascending columns inside each segment; weights already follow that order.
No ``edge_delays`` array is needed or retained. The scheduling counts target
the current hybrid thresholds and must be regenerated if those change.
``buffer_slots`` defaults to the maximum delay plus one. Optional ``ring``,
``head`` and mode-appropriate ``workspace`` restore an existing execution
state; omitted buffers are freshly allocated. JAX operands are reused and
NumPy operands receive owned snapshots.

The default ``validate='static'`` checks shapes, dtypes and host parameters
without reading connectivity or metadata contents. The existing head range
check still applies. ``validate='full'`` additionally reads concrete layout
arrays on the host to check CSR bounds, segment counts and column ordering,
delay offsets, row tasks and worst-case capacity. It does not rebuild or sort
anything, and cannot verify original edge-delay labels without per-edge
delays. Static construction supports JIT and differentiation; full validation
requires concrete layout arrays. Both modes retain primitive runtime checks.

Both backends share input validation at the primitive entry. Shapes and dtypes
are checked during tracing; dynamic head, CSR/delay metadata and active-task
capacity are checked with JAX before delivery. Metadata is checked even for
inactive rows. With the raw primitive's ``check_status=False``, invalid metadata
or capacity is reported in status without delivering into the ring; invalid
heads always raise. Consuming only status or a nonempty task array retains the
check. Eager and JIT calls use the same execution-time checks; an asynchronous
error can surface when the result is synchronized.

The CUDA FFI invokes the prepared delivery core directly. Independent C++
``launch_*`` calls use a shared native preflight before invoking that same core.
The delivery kernels do not repeat head, metadata or capacity validation.
Assigning ``delaycsr.head`` also supports clearing or reading without delivery,
so assignment checks its scalar integer and range contract, using the same
Python range predicate and checking wide integers before conversion to int32.

``clear_head`` clears the current slot with JAX. ``forward`` and ``x @ W``
use the same clearing operation before additive delivery and return the next
slot. For JAX transformations, pass the state explicitly and return its updated
pytree; use ``with_data`` for a local weight-replacement copy when taking
gradients. JIT-compiling a bound method or mutating a closed-over state is
unsupported and is not diagnosed by these methods.

Events may use boolean, integer or floating dtypes. Binary delivery uses
``event != 0``; weighted delivery also uses the event amplitude. The ring's
mantissa dtype determines execution storage: ``float16``, ``bfloat16``,
``float32`` or ``float64``. Weights are converted temporarily to that dtype
for delivery, and the weighted task workspace uses the same dtype. An
explicit ring may have a different dtype from the stored weights.

Weights and events may carry units. The ring must have units compatible with
their product. Delivery converts contributions to the ring's unit and preserves
the ring's dtype and unit, so its pytree structure stays fixed across
``scan`` and ``while_loop`` steps. Unit scaling uses at least float32 before
conversion to the ring dtype; stored arrivals are not rescaled. Supply a
compatible ring when using unitful events.
Canonical weights retain their original dtype and unit. Construction and
weight replacement copy caller-owned NumPy storage.

Weight replacements via ``with_data`` must follow the initialized
``delaycsr.data`` edge order and match its shape, dtype and unit exactly.
Shared weights retain shape ``(1,)`` and per-edge weights retain shape
``(nnz,)``. Changing the execution ring dtype does not relax these rules.

Event activity and derivative conventions follow ``main@c099629``, including its
weighted-event underflow difference. JAX checks activity on the original
event, then casts its amplitude to the ring dtype. CUDA casts weighted
events before its native activity check. Consequently, a nonzero event that
rounds to zero can propagate NaN against an infinite weight on JAX while
contributing zero on CUDA. Event JVPs retain this backend difference;
weight VJPs use original event activity on both backends. Ordinary zero
events still contribute exactly zero against NaN or infinite weights.

.. autosummary::
   :toctree: generated/

   delaycsr
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
