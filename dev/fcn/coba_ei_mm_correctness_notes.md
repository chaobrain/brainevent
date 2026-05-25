# COBA EI MM correctness notes

## Context

`coba_ei_benchmark_mm_test.py` compares COBA EI matrix-matrix routes against the
`binary` + `jax_raw` reference. The direct small-shape route test exposed a
stable mismatch for:

- `data_type='compact'`
- `efferent_target='post'`
- `mv_layout='row_gather'`
- both homogeneous and heterogeneous weights

## Root cause

The post-synaptic MM route evaluates `operand @ conn`, so 2D compact input uses
`FixedPostNumConn.__rmatmul__`.

The old compact branch transposed only `other.value` before calling
`compact_binary_fcnmm(..., transpose=True)`, but reused the original
`other.packed`, `other.active_ids`, and `other.n_active`. For `CompactBinary`,
these buffers describe active rows of the original matrix. After
`other.value.T`, the compact metadata no longer matched the matrix consumed by
the FCNMM kernel.

The fix rebuilds the compact representation from `other.value.T` before calling
`compact_binary_fcnmm`. The same pattern is applied to the corresponding
`FixedPreNumConn.__rmatmul__` 2D compact branch.

## Test helper fix

The MM direct route test now uses `preferred_real_mm_backend(data_type)` instead
of the MV-oriented `preferred_real_backend(data_type)`. This is not the root
cause of the compact dispatch bug, but it keeps MM tests selecting from MM
primitive backends.

## Large-scale trace checking

The previous large-scale test compared only spike history. That could pass even
when a connection operator was wrong, because the fake neuron path can hide
conductance differences.

The large-scale helper now records exact spike bits plus per-step intermediate
summaries:

- full spike trace packed with `jnp.packbits`
- excitatory and inhibitory connection delta summaries
- excitatory and inhibitory synaptic conductance summaries
- input current summary

It still drives the network with the same per-step call pattern:

```python
net.update(t, 24 * u.mA, exc_conn=exc_conn, inh_conn=inh_conn)
```

Spike equality is exact: actual and reference traces must have the same packed
bits and the same original spike-trace shape. For memory safety, the other
large-scale intermediates compare summaries rather than returning full tensors.
Each summary includes:

- mean
- mean absolute value
- max
- min
- nonzero ratio
- 32 fixed-position samples across the flattened tensor

To make the connection path active during the fake-network trace, the helper
initializes a deterministic nonzero spike pattern before running the traced
steps. Actual and reference networks receive the same initial spike pattern.

## MV large-scale alignment

The MV large-scale test uses the same step-trace comparison as MM, but keeps a
separate memory-safe point. The MM large-scale point intentionally lowers
`scale` and raises `conn_num` to stress high-connectivity MM behavior.

Current MV large-scale point:

- `scale=798`
- `conn_num=20`

Current MM large-scale point:

- VRAM estimate input: `limit_gb=1`, `batch_size=32`, `scale=100`
- Generated fixed connection count: `607`

This keeps MV from OOMing on high-connection CSC/connection construction while
still exercising the MV route with step-trace comparison.

## Current MM large-scale coverage

`test_large_scale_mm_e2e_spikes_match_binary_jax_reference` currently checks:

- `batch_size=32`
- `scale=100`
- `conn_num=607`
- `steps=10`
- `homo=True`
- actual input current of `24 * u.mA` at every step
- actual backend selected by `preferred_real_mm_backend(data_type)`
- reference backend fixed to `binary` + `jax_raw`
- all 12 `ROUTE_CASES`

The covered route cases are:

- `binary`: `post/row_gather`, `pre/row_gather`, `pre/col_scatter`
- `compact`: `post/row_gather`, `pre/row_gather`, `pre/col_scatter`
- `bitpack`: `post/row_gather`, `pre/row_gather`
- `bitpack_a0`: `post/row_gather`, `pre/row_gather`
- `bitpack_a1`: `post/row_gather`, `pre/row_gather`

The test has two skip gates:

- the whole e2e large-scale test is skipped if binary `jax_raw` is unavailable
- a per-route case is skipped if `preferred_real_mm_backend(data_type)` returns
  `None`

On the current GPU environment, binary `jax_raw` is available and every route
selects `cuda_raw` as the real MM backend, so no MM large-scale case is skipped.

Known remaining gaps:

- large-scale non-spike intermediates use summaries, not full tensors
- large-scale MM currently uses `homo=True` only
- `float` FCN routes are not part of the COBA EI binary/compact/bitpack route
  matrix

## Large-scale coverage recheck

The MM large-scale e2e test should not be read as exhaustive coverage of every
FCNMM operator and every dispatch path.

It does cover the 12 COBA EI binary-family route cases at large scale, with no
skips on the current GPU environment:

- `binary`, `compact`, `bitpack`, `bitpack_a0`, and `bitpack_a1`
- COBA EI `post` routes through `operand @ conn`
- COBA EI `pre` routes through `conn @ operand`
- binary/compact `pre/col_scatter` construction with maintained dual layout

However, the current large-scale batch helper uses `brainstate.nn.Map` over a
single-neuron-network update. Instrumentation of the Python dispatch entry
shows the network update enters the 1D `*_fcnmv` branches:

- `post`: `FixedPostNumConn.__rmatmul__` with 1D binary-family operands
- `pre`: `FixedPostNumConn.__matmul__` with 1D binary-family operands

So the large-scale e2e test validates the batched COBA EI network behavior and
the mapped MV dispatch path. It is not, by itself, proof that every explicit
2D `*_fcnmm` branch in `FixedPostNumConn.__matmul__` and
`FixedPostNumConn.__rmatmul__` has been exercised at large scale. Direct
`jax.vmap` over these MV calls can promote the primitive to `*_fcnmm`, but the
large-scale COBA EI test is still not a replacement for explicit 2D MM route
tests.

Explicit 2D MM coverage currently comes from the small-shape route test:

- `test_each_route_mm_matches_binary_jax_reference_on_small_shapes`
- both `homo=True` and `homo=False`
- all 12 `ROUTE_CASES`
- actual 2D operands, so the 2D `binary_fcnmm`, `compact_binary_fcnmm`, and
  `bitpack_binary_fcnmm` branches are selected
- `test_bitpack_specialized_mm_backends_match_binary_jax_reference_on_small_shapes`
  covers the bitpack FCNMM specialized backends:
  `bit_full_warp`, `bit_full_block`, `bit_basic_only`, `bit_warp_only`,
  `bit_full_warp_a0`, and `bit_full_warp_conn_u8`

Still not covered by MM large-scale:

- `float`/dense `fcnmm`
- `compact_only_vector`
- `mv_layout='auto'`
- `homo=False`
- `FixedPreNumConn` MM paths, because the COBA EI benchmark builds
  `FixedPostNumConn` even for `efferent_target='pre'`
- dtype-specific kernel variants beyond the default dtype used by the test
- full tensor equality for non-spike intermediates at large scale

## `test_colmajor_fullwarp_nocap` participation recheck

Rechecked on 2026-05-23 with `/home/luruheng/miniconda3/bin/python`,
JAX/JAXLIB 0.9.0.1, default backend `gpu`, and device `CudaDevice(id=0)`.

Targeted verification:

- `fcnmm_testing_op_test.py` plus the targeted binary/COBA EI MM cases:
  `32 passed, 1 skipped`
- `test_small_scale_mm_e2e_spikes_match_binary_jax_reference[binary-post-row_gather]`
  and
  `test_large_scale_mm_e2e_spikes_match_binary_jax_reference[binary-post-row_gather]`:
  `2 passed`
- `test_binary_fcnmm_test_colmajor_nocap_fixed_post_batch_route_matches_reference`
  and `test_binary_fcnmv_batched_global_test_colmajor_backend_matches_reference`:
  `2 passed`

The important distinction is between two ways of selecting the backend:

- `dev/fcn/COBA_EI_binary_fcnmm_CsvOuput.py` selects the backend globally with
  `brainevent.config.set_backend(runtime_platform, back)`, then calls
  `make_simulation_batch_run(...)` without passing the `backend` keyword. In
  this path the `FixedPostNumConn` instances have `conn.backend is None`.
- The earlier debug probe passed
  `backend='test_colmajor_fullwarp_nocap'` into `make_simulation_batch_run`.
  That stores the backend on each connection object. This is not the same path
  as the CSV benchmark script.

The CSV benchmark path does participate in the test-colmajor FCNMM backend:

- The public 2D route participates in `binary_fcnmm`.
  `BinaryArray(spikes_2d) @ FixedPostNumConn(...)` calls
  `binary_fcnmm_p_call` with `matrix_shape=(8, 4)` and `transpose=True`.
  With the global GPU backend set to `test_colmajor_fullwarp_nocap`, that
  dispatches to the test-colmajor FCNMM primitive backend.
- `make_simulation_batch_run(...)` under the global backend also starts from a
  mapped 1D `binary_fcnmv` call, but because the primitive call has
  `backend=None`, the `binary_fcnmv` batching rule rewrites the batched call to
  `binary_fcnmm_p_call(..., backend=None)`. The global backend is then resolved
  by `binary_fcnmm`, where `test_colmajor_fullwarp_nocap` is registered.
- Lowered StableHLO for the CSV-style path contains this custom call:

```text
stablehlo.custom_call
  @fcn_fcnmm_testing.binary_fcnmm_test_colmajor_fullwarp_nocap_homo_bool_f32
```

So the Nsight result showing
`_test_fcnmm_colmajor_fullwarp_homo_kern_bool_f32` is consistent with
`dev/fcn/COBA_EI_binary_fcnmm_CsvOuput.py`.

The failed path was the full mapped simulation debug/helper variant with the
backend stored on the connection object. In that `brainstate.nn.Map` path the
traced update first presents each batch member as a 1D spike vector, so the
primitive is `binary_fcnmv` with an explicit backend. `test_colmajor_fullwarp_nocap`
is not registered for `binary_fcnmv`:

```text
KernelFallbackExhaustedError:
test_colmajor_fullwarp_nocap not available for platform gpu in primitive binary_fcnmv.
```

The dispatch trace from that failed explicit-backend attempt is:

```text
CALL ('prim_mv', 'test_colmajor_fullwarp_nocap', (32,), 1, True)
CALL ('prim_mv', 'test_colmajor_fullwarp_nocap', (8,), 1, True)
```

This does not mean every explicit-backend call fails. A direct public 2D call
such as `BinaryArray(spikes_2d) @ conn` still enters `binary_fcnmm` and lowers
successfully even when `conn.backend == 'test_colmajor_fullwarp_nocap'`.

The matching CSV-style run succeeds with `conn.backend is None`; Python-level
instrumentation sees the original mapped MV call before batching, and lowered
IR confirms that the compiled program contains the FCNMM test-colmajor custom
call:

```text
CALL ('api_mv', None, (32,), 1)
CALL ('prim_mv', None, (32,), 1, True)
CALL ('api_mv', None, (8,), 1)
CALL ('prim_mv', None, (8,), 1, True)
```

### Direct 2D MM spike dump

This is the small direct route used to prove the backend itself participates in
`binary_fcnmm`. The public spike matrix is laid out as `[batch, pre]`:

```text
[[0 1 0 0 0 0 1 0]
 [1 0 0 0 0 1 0 1]
 [0 0 0 0 1 0 1 0]
 [0 0 0 1 0 1 0 0]]
```

The matrix passed to the FCNMM primitive is the transpose, `[pre, batch]`.
Both `test_colmajor_fullwarp_nocap` and `jax_raw` receive this same spike
content:

```text
test_colmajor_fullwarp_nocap spike operand:
[[0 1 0 0]
 [1 0 0 0]
 [0 0 0 0]
 [0 0 0 1]
 [0 0 1 0]
 [0 1 0 1]
 [1 0 1 0]
 [0 1 0 0]]

jax_raw reference spike operand:
[[0 1 0 0]
 [1 0 0 0]
 [0 0 0 0]
 [0 0 0 1]
 [0 0 1 0]
 [0 1 0 1]
 [1 0 1 0]
 [0 1 0 0]]
```

The FCNMM outputs matched exactly, with `max_abs_diff=0.0`:

```text
test_colmajor_fullwarp_nocap output:
[[1.25 1.25 1.25 2.5 ]
 [1.25 1.25 1.25 0.  ]
 [1.25 1.25 0.   1.25]
 [1.25 2.5  2.5  1.25]
 [1.25 1.25 0.   0.  ]
 [1.25 1.25 2.5  2.5 ]
 [0.   1.25 0.   0.  ]
 [1.25 1.25 1.25 1.25]
 [1.25 2.5  1.25 1.25]
 [1.25 1.25 0.   0.  ]
 [1.25 2.5  2.5  2.5 ]
 [0.   1.25 0.   0.  ]]

jax_raw reference output:
[[1.25 1.25 1.25 2.5 ]
 [1.25 1.25 1.25 0.  ]
 [1.25 1.25 0.   1.25]
 [1.25 2.5  2.5  1.25]
 [1.25 1.25 0.   0.  ]
 [1.25 1.25 2.5  2.5 ]
 [0.   1.25 0.   0.  ]
 [1.25 1.25 1.25 1.25]
 [1.25 2.5  1.25 1.25]
 [1.25 1.25 0.   0.  ]
 [1.25 2.5  2.5  2.5 ]
 [0.   1.25 0.   0.  ]]
```

## Additional bitpack backend finding

Adding the specialized bitpack FCNMM backend tests exposed a real bug in
`bit_full_warp_conn_u8` for homogeneous weights. Its staged connection-loop
kernel assigned `output = w0` instead of accumulating contributions. That lost
updates when multiple active connections targeted the same output entry. The
kernel now uses the same atomic-add pattern as the other scatter kernels.

## Verification run

Verified targeted paths after the compact MM dispatch fix:

- `test_each_route_mm_matches_binary_jax_reference_on_small_shapes`: 24 passed
- large-scale `binary-post-row_gather`: passed
- large-scale `compact-post-row_gather`: passed
- large-scale `bitpack_a1-post-row_gather`: passed

Verified after MV large-scale alignment:

- `test_large_scale_mv_e2e_spikes_match_binary_jax_reference` plus
  `test_large_scale_mm_e2e_spikes_match_binary_jax_reference`: 24 passed,
  0 skipped
- full MV correctness file: 83 passed, 0 skipped
- full MM correctness file: 48 passed, 0 skipped
- `test_batch_step_trace_contains_full_packed_spike_bits`: passed
- `test_bitpack_specialized_mm_backends_match_binary_jax_reference_on_small_shapes`:
  12 passed
