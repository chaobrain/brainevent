# Differentiable plasticity primitives and generic per-edge rules

Issue: #202. Branch: `feat/plasticity-ad`.

## Problem

1. The event-driven STDP primitives `binary_csr_plast` (`update_csr_on_binary_pre`),
   `csr2csc_on_post` (`update_csr_on_binary_post`) and `fcn_plasticity_row`
   (`update_fixed_post_conn_on_binary_pre`, `update_fixed_pre_conn_on_binary_post`) have no
   JVP / transpose rules: `jax.grad` raises `NotImplementedError: Differentiation rule for
   'binary_csr_plast' not implemented`. The dense primitives differentiate only w.r.t. the
   weight; a trace tangent raises.
2. The update is fixed to "add trace, clip". There is no per-edge state besides the weight
   and no weight-dependent or multiplicative term.

## Part (a): AD rules for the existing primitives

Every primitive computes, for a fixed spike vector `s` and structure,

```
out[e] = w[e] + m[e] * trace[t(e)],      m[e] = event_active(s[d(e)])  in {0, 1}
```

where `d(e)` is the driving neuron (pre row for `on_pre`, post column for `on_post`) and `t(e)`
the other endpoint. The map is **jointly linear in `(weight, trace)`**. Spikes and structure
(`indices`, `indptr`, `weight_indices`) are non-differentiable: their JVP slot is `None` and
the transpose rule raises if asked for their cotangent. Clipping to `[w_min, w_max]` stays in
the Python wrapper (`u.math.clip`), so `w_min` / `w_max` gradients come from JAX; nothing is
added for them.

| primitive | weight JVP | trace JVP | weight cotangent | trace cotangent |
|---|---|---|---|---|
| `binary_csr_plast` | `w_dot` | `prim(0, idx, ptr, s, trace_dot)` | `ct` | `segment_sum(m[e] ct[e], indices[e], n_post)` |
| `csr2csc_on_post` | `w_dot` | `prim(0, csc_idx, csc_ptr, perm, trace_dot, s)` | `ct` | `segment_sum(m[k] ct[perm[k]], csc_idx[k], n_pre)` |
| `fcn_plasticity_row` | `w_dot` | `prim(0, idx, s, trace_dot)` | `ct` | `zeros(n).at[idx].add(m[r] ct[r, k])` |
| `dense_on_pre` | `w_dot` | `prim(0, s, trace_dot)` | `ct` | `event_factor(s) @ ct` |
| `dense_on_post` | `w_dot` | `prim(0, trace_dot, s)` | `ct` | `ct @ event_factor(s)` |

Decisions:

- **Trace JVP reuses the primitive** with a zero weight. It is event-driven, runs on the same
  `backend=` as the primal (the rule forwards `backend`), and is linear in `trace_dot`, so its
  transpose is the trace branch of the same transpose rule.
- **Trace cotangent is pure JAX** (gather + masked `segment_sum`, O(nse)). The event-driven
  alternative `binary_csrmv(ct, ..., transpose=True)` needs a host-built task workspace from a
  *concrete* `indptr` (`hybrid_task_capacity`), which is not available inside a transpose rule
  where the structure may be traced. The cotangent dtype is cast to the trace dtype.
- The cotangent mask is `event_active(s)` (bool → `x`, numeric → `x != 0`), matching the
  forward activity contract (`2026-10-24-event-activity-and-abi.md`): negative and NaN spikes
  are active.
- **Batching:** the default `general_batching_rule` (a `lax.scan` over the batch) remains for
  the CSR / FCN primitives; dense keeps its existing `(0, None, None)` fast path. The JVP
  primitives batch through the same rule and the transpose rules are pure JAX, so `vmap` of
  `grad` / `jvp` composes.
- The CSC wrappers (`update_csc_on_binary_pre/post`) and the FCN unfavourable directions reuse
  these primitives and become differentiable with no further code.

## Part (b): generic per-edge rule API

New module `brainevent/_csr/plasticity_generic.py`, pure JAX (differentiable through `rule`,
`edge_state`, `pre_vals`, `post_vals` and `params`):

```python
csr_apply_on_pre(rule, edge_state, indices, indptr, pre_spike, *,
           pre_vals=None, post_vals=None, params=None, shape)
csr_apply_on_post(rule, edge_state, indices, indptr, post_spike, *,
            pre_vals=None, post_vals=None, params=None, shape)
```

- **Naming.** The issue proposed `csr_on_pre`, but `brainevent.csr_on_pre` is a deprecated
  alias of `update_csr_on_binary_pre` (`_deprecation.DEPRECATED_RENAMES`) with a different
  signature; reusing it would silently change the meaning of old code. The functions are named
  `csr_apply_on_pre` / `csr_apply_on_post`, matching the `CSR.apply_on_pre/post` methods.
- `indices`, `indptr`: the CSR structure, `shape = (n_pre, n_post)`.
- `edge_state`: pytree whose leaves have leading dimension `nse` (CSR data order). Trailing
  dimensions are allowed.
- `pre_vals` / `post_vals`: pytrees of leading dimension `n_pre` / `n_post`, gathered to edges
  with the edge's row id / `indices`.
- `params`: any pytree, passed unchanged (broadcast) to every call.
- `rule(edge_e, pre_e, post_e, params) -> new_edge_e` sees one edge (leaves with the leading
  dimension removed) and must return a pytree with the structure of `edge_state`. It is applied
  with `jax.vmap` over all edges.
- Result leaf: `where(active[e], new, old)`, cast back to the dtype of the `edge_state` leaf so
  it can be a `lax.scan` carry. `active[e]` is `event_active(pre_spike[row(e)])` for
  `csr_apply_on_pre` and `event_active(post_spike[indices[e]])` for `csr_apply_on_post`.
- **No `weight_indices` argument.** The post id of every CSR edge is `indices[e]`, so the post
  activity is gathered per edge directly; a CSC permutation is only needed by kernels that
  iterate over post columns. The API stays minimal; a fused CSC-iterating kernel can add it
  later without breaking callers.
- Errors: `TypeError` when the rule output structure differs from `edge_state`; `ValueError`
  for leaves whose leading dimension does not match `nse`, `n_pre` or `n_post`, or when the
  spike vector length does not match `shape`.
- Units: leaves may be `brainunit.Quantity`; selection uses `u.math.where`.

Cost: the implementation evaluates `rule` on every edge and masks (O(nse) per call) — the same
work as brainpy.state's reference path, now as a library call. A fused event-driven kernel
(evaluate `rule` only on edges of active neurons) is future work; the API is designed so it can
be swapped in.

Gradient caveat (documented): `where` sends a zero cotangent to the rule on inactive edges; if
the rule's derivative is non-finite there (e.g. `log(0)`), `0 * inf = nan` propagates, as with
any `jnp.where`.

### `CSR` methods

```python
CSR.apply_on_pre(rule, pre_spike, *, edge_state=None, pre_vals=None, post_vals=None, params=None)
CSR.apply_on_post(rule, post_spike, *, edge_state=None, pre_vals=None, post_vals=None, params=None)
```

They call `csr_apply_on_pre/post` with the matrix structure and shape. `edge_state=None` means
`self.data` (the rule then receives a scalar weight). They always **return the new edge state**
(the new data array when `edge_state` is `None`); use `csr.with_data(...)` to rebuild a matrix.
Returning one type keeps the method predictable for multi-state rules. Homogeneous (size-1)
data is rejected like the binary updates.

### Exports and docs

- `brainevent._csr.__init__`, `brainevent.__init__`: `csr_apply_on_pre`, `csr_apply_on_post`.
- `docs/reference/apis/operations.rst`: CSR "Plasticity operations".
- `docs/how-to/data-structures/synaptic-plasticity.rst`: a weight-dependent STDP rule with an
  eligibility trace, and a gradient through a rule parameter; note that the binary operators
  are differentiable.

## Tests

- `_csr/plasticity_binary_test.py`, `_dense/plasticity_binary_test.py`,
  `_fcn/plasticity_binary_test.py`: on every backend of `<prim>.available_backends(platform)`,
  spikes containing `2, -1, 0`: forward, `jax.grad` w.r.t. weight, trace, `w_min`, `w_max`
  vs a pure-JAX reference; `jax.jvp` vs reference; JVP/VJP dot-product identity; `vmap` of
  grads; the exact #202 repro; CSC variants; FCN both directions.
- `_csr/plasticity_generic_test.py`: weight-dependent STDP + eligibility rule vs a numpy loop
  (pre and post); multi-leaf edge state with trailing dims; grads through `params` and
  `edge_state` vs a dense reference; `jit` and `vmap`; inactive edges bit-identical; error
  paths; `CSR` methods; units.
