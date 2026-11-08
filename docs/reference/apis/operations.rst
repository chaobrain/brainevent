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

``apply(fn)`` has the same shape, dtype and exact unit requirements, including
the unit's scale. It returns a copy with the existing delay layout and runtime
state. ``apply2(other, fn, reverse=False)`` also preserves the weight dtype and
unit, but permits either shared ``(1,)`` or per-edge ``(nnz,)`` results. Its
result uses the receiver object's ring, head and workspace; ``reverse=True`` only
reverses the arithmetic operands. Compare weight units with weight units:
the ring may have a different unit or precision.

A sparse operand to ``apply2`` or arithmetic must be another ``delaycsr``
with the same shape, ring length, delay steps and weighted mode. The two
objects must share the same ``indices``, ``indptr``, ``delay_count`` and
``delay_off`` array objects. Numerically equal, separately allocated layouts
are rejected because equality alone does not establish the supported shared
layout contract. Runtime states need not match. Plain CSR/CSC operands have
no delay layout and are rejected; pass explicit weight values for value-only
operations. Addition or subtraction that returns a dense array still has the
parent's static weight semantics.

``delaycsr.fromdense(mat, *, template=W)`` uses an explicit existing layout
to replace weights. It reads each dense value in the template's current edge
order and produces per-edge weights, including when the template uses one
shared weight. The result preserves the template's delay metadata, config,
structure caches and runtime state. Previously accumulated arrivals keep
their old values; new weights affect future deliveries. Shape, dtype and
exact weight unit must match the template. Existing edges assigned zero remain
stored edges with their existing delays. Nonzero values outside the template
and templates containing duplicate coordinates raise ``ValueError``: a dense
array cannot specify new edge delays or separate weights for parallel edges.
Calling the classmethod through an instance does not supply ``template``
implicitly.

``fromdense`` emits ``UserWarning`` because changing connectivity rather than
weights invalidates the old edge-to-delay correspondence. For a structural
change, construct a new layout with explicit per-edge delays. This validation
requires concrete arrays outside JIT; use ``with_data`` for weight replacement
inside JIT or AD.

``slice_rows(index)`` selects complete pre rows, including their delay counts
and task counts, and returns a new ``delaycsr``. It preserves selection order,
repeated rows, within-row edge order, output columns, ring length, weight dtype
and unit, and shared-weight storage. It rebuilds row pointers, capacity,
structure caches and config for the selected structure, and allocates new
workspace. Its ring is zero and its head is zero; the source is unchanged.
Aggregated arrivals cannot be separated into contributions from selected pre
rows. The new ring retains the source ring's own dtype and unit.

``diag_add(other, *, diag_delays)`` requires a one-dimensional increment array
of length ``min(W.shape)`` and explicit delays for those increments. Delays
must be a nonnegative integer scalar or an integer vector of that length.
Each increment becomes a separate diagonal edge, even if an edge with the
same coordinate and delay already exists or the increment is zero. Existing
edges retain their weights and delays. For example, an existing weight 2 at
delay 1 and an increment 3 at delay 4 remain two edges; they do not become
weight 5 at delay 4. Compatible weight units are converted and weight dtypes
are promoted according to normal construction rules.

``diag_add`` reconstructs through the ordinary initializer, sorts the new
layout, regenerates config and task metadata, and allocates fresh runtime
state. It extends the ring to ``max(old_L, max(diag_delays) + 1)`` when the
diagonal is nonempty, resets head to zero, and clears arrivals. The new ring
preserves the original ring's dtype and unit; the source is unchanged. Config
follows the initializer's rules rather than inheriting old tuning. This method
emits ``PerformanceWarning`` for host reads, weight expansion, sorting and
allocation. Both structural operations require concrete inputs outside JIT;
``diag_add`` is not intended for every simulation timestep.

The following example uses a template whose row order already follows its
delay layout. ``fromdense`` and ``diag_add`` emit the warnings described above.

.. code-block:: python

    >>> import numpy as np
    >>> from brainevent import delaycsr
    >>> matrix = delaycsr(
    ...     (np.array([2., 3.], np.float32), np.array([1, 0], np.int32),
    ...      np.array([0, 1, 2], np.int32)),
    ...     shape=(2, 2), edge_delays=np.array([1, 2], np.int32),
    ...     ring=np.ones((3, 2), np.float32), head=1, backend='jax_raw')
    >>> updated = delaycsr.fromdense(
    ...     np.array([[0., 5.], [7., 0.]], np.float32), template=matrix)
    >>> np.asarray(updated.data)
    array([5., 7.], dtype=float32)
    >>> int(updated.head), bool(np.array_equal(updated.ring, matrix.ring))
    (1, True)
    >>> selected = matrix.slice_rows([1, 0, 1])
    >>> selected.shape, int(selected.head), bool(np.all(selected.ring == 0))
    ((3, 2), 0, True)
    >>> expanded = matrix.diag_add(
    ...     np.array([1., 4.], np.float32), diag_delays=np.array([3, 0], np.int32))
    >>> expanded.delay_steps, expanded.L, expanded.indices.size
    ((0, 1, 2, 3), 4, 4)

``build_weight_indices()`` prepares the parent CSR's auxiliary CSC indices
and edge permutation and returns a plain ``CSR``. It emits ``UserWarning``:
the returned object is for static weight and structure extraction and has no
delay metadata, ring, head or delay workspace. The source delay state is
unchanged. This export is distinct from ``tocsr()``, which returns the original
``delaycsr`` itself.

``W @ x``, ``W[index]`` and ``prepare_diag_add()`` explicitly raise
``NotImplementedError``. Use ``events @ W`` or ``W.forward(events, t)`` for
delayed delivery, and ``slice_rows`` for selecting pre rows. For static dense
indexing, ``W.todense()[index]`` materializes the full dense weight matrix.
``prepare_diag_add`` cannot preplan the rebuilding ``diag_add`` for JIT.

The inherited plasticity methods ``update_on_pre``, ``update_on_post``,
``apply_on_pre`` and ``apply_on_post``, and the mapping methods ``dt2t`` and
``dt2t_transposed``, reject original int64 ``indices`` with ``TypeError`` on
every backend before preparing caches or running the parent calculation.
They do not cast indices silently. This restriction does not ban int64
``indptr`` or int64 indices in native delayed delivery; pointer support still
depends on the underlying operation. Deprecated ``yw_to_w`` aliases use the
same checks. The update methods return a copied ``delaycsr``; the apply methods
return edge-state arrays or pytrees. Signals are used as supplied, with no
automatic alignment to each edge's delayed history. Explicit edge state must
follow the layout's current sorted edge order.

Inherited static operations retain the following boundaries. Delay delivery
uses its own metadata and does not require prepared ``CSRConfig`` data;
inherited static operations may have their own config requirements.

.. list-table:: Static weight operations and delay state
   :header-rows: 1
   :widths: 28 72

   * - Operation
     - Meaning for ``delaycsr``
   * - ``sum``, ``todense``, basic attributes
     - Query static weights or structure, rather than current arrivals.
   * - ``tocsc``, ``tocoo``, ``transpose``, ``T``
     - Return ordinary static matrices without delay or runtime state.
       ``W.T @ x`` is an instantaneous CSC product.
   * - Dense results from ``+`` or ``-``
     - Contain static weight values rather than arrival-ring values.
   * - ``solve``
     - Solves the static weight system under the parent's dtype and backend
       requirements; it does not solve delayed dynamics.
   * - ``dt2t``, ``dt2t_transposed``
     - Map static edge quantities; they are not gradients of the complete
       delayed system.
   * - ``delay_layout``
     - Reinitializes a layout using delays in the current edge order, with a
       fresh ring and head and default ``weighted=False``. It is not a state
       preserving clone.
   * - ``with_rebuilt_config``
     - Still returns a plain ``CSR`` and loses delay/runtime information.
       Subtype preservation for this inherited method remains unresolved.

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
