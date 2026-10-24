# `WeightedArray`: value-weighted event representation (#201)

Branch: `feat/weighted-array` (stacked on `fix/event-abi-activity`).

## Motivation

Spiking simulators need count-valued or graded events: a source can emit several spikes in
one step (NEST `poisson_generator` multiplicity, adaptive neurons firing several times per
step, merged populations), and the required product is `y[j] = Σ_i x[i] W[i, j]` with most
`x[i] = 0`. `BinaryArray` binarises (`x != 0` → 1); a float matmul is exact but not
event-driven. brainpy.state currently uses the float matmul for every source with
multiplicity > 1.

## Contract

```python
brainevent.WeightedArray(value)       # value: int8/uint8/int16/int32/float16/bfloat16/float32/float64
```

- **Forward.** `WeightedArray(x) @ M` (and `M @ WeightedArray(x)`) computes the exact product
  `x @ dense(M)` while visiting only rows/columns with `x != 0`. Signed values are allowed.
- **Derivatives.** Exact, the derivatives of the linear map:
  - weight JVP: the same weighted primitive applied to the weight tangent;
  - weight cotangent: `dW[e] = x[pre(e)] · ct[post(e)]` (an SDDMM restricted to active rows);
    homogeneous / JITC scalar weights: `<ct, prim(ones)>`;
  - event cotangent / JVP: `W^T ct` / `prim(W, x_dot)` (float op), identical to the binary
    straight-through formula. Integer events get `float0` cotangents (JAX convention).
- **Not a `BinaryArray` subclass.** Every existing `isinstance(x, BinaryArray)` stays binary.
  Matrix classes dispatch through `brainevent._event.as_event_operand(x) -> (value, weighted)`
  (or `None` for non-event operands). A format that cannot handle a `WeightedArray` raises
  `TypeError`; it never silently densifies or binarises.
- **Formats.** Event-driven for all formats:

  | format | implementation |
  |---|---|
  | Dense, CSR/CSC (+ indexed route), FixedNumPerPre/Post, TCSR | static `weighted: bool = False` parameter on the existing binary primitives (`binary_densem{v,m}`, `binary_csrm{v,m}`, `binary_csrm{v,m}_indexed`, `binary_fcnm{v,m}`, TCSR `binary_csrm{v,m}`); kernels multiply by the event value in the active branch; rules use `w * x` instead of `w * mask(x)` |
  | JITCScalar/Normal/Uniform R/C | route to the existing float JITC primitives (`jitsmv/jitnmv/jitumv` and `mm`), whose kernels skip zero events (add the skip where missing: uniform) and use the same RNG chunking as the binary kernels, hence the same matrix |
  | cuSPARSE / cuBLAS backends | float op |

- **CUDA ABI.** `cuda_event_abi(dtype, weighted=True, value_dtype=W.dtype)` returns
  `EventABI('_wv', cast-to-value_dtype)`: weighted kernels read the event buffer in the weight
  dtype (one extra instantiation per weight dtype × homo/hetero per kernel); integer events
  are cast inside the lowering. TCSR weighted mm uses the indexed hybrid path (the tile path
  only consumes active-row lists and stays binary-only). TCSR weighted gradients use the
  value-weighted (`pp_prop`) weight cotangent.
- **Activity in weighted kernels** is `x != 0` (same as binary); `0 * w` terms are skipped,
  so a NaN/Inf weight in an inactive row does not propagate.

## Helpers

- `brainevent._op.util.event_factor(x, *, dtype, weighted=False)`: mask (binary) or
  `x.astype(dtype)` (weighted).
- `brainevent._op.util.cuda_event_abi(..., weighted=False, value_dtype=None)`.
- `brainevent._event.as_event_operand(x)`.

## Tests

`check_event_op_grads(op, dense_op, params, x, mode='weighted')` for every format, mv and mm,
both transpose directions, homo/hetero, every backend, with `x` containing `2, -1, 0, 0.5`
in int8/uint8/int16/int32/f16/bf16/f32/f64. Binary results unchanged (existing tests). GPU:
compile time of the added instantiations and binary-path benchmarks before/after (no
regression).
