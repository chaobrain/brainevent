# Event activity semantics and CUDA event ABI

Issues: #199, #204 (and the #200 regression test). Branch: `fix/event-abi-activity`.

## Problem

1. **Inconsistent activity.** A binary event is "active" under `x > 0` in most kernels
   (CUDA `IS_ACTIVE_FLOAT*`, CSR/FCN/JIT numba and jax), under `x != 0` in the plasticity
   CPU/jax kernels, compaction, `bitpack` and integer casts, and is not thresholded at all in
   the dense `jax_raw` kernels (raw float values are multiplied in). The same input gives
   different results on different backends.
2. **Wrong CUDA event ABI (#199, #204).** Most CUDA generators pick the event symbol with
   `'_bool' if dtype == bool else '_float'`, where `_float` means a float32 buffer. float64,
   float16, bfloat16 and integer events reach a float32 kernel: silent garbage (dense mv/mm,
   #204), out-of-bounds reads, or `std::abort()` in the hybrid CSR kernels (#199). Commit
   e18f81c fixed only CSR binary mv/mm.
3. **`abort()` in host code.** Hybrid dtype checks, `dispatch.h` default branches, index
   checks, JIT grid guards and cuBLAS status checks call `abort()`, killing the Python process
   instead of raising.

## Contract

- **Activity.** A numeric event `x` is active iff `x != 0`; a boolean event iff `x`. This holds
  in every backend (numba, jax_raw, pallas, cuda_raw, cusparse, cublas) and every operation
  (matmul, plasticity, compaction, bit-packing). Consequences: negative values and NaN are
  active. Binary operations never multiply by the event value.
- **CUDA event ABI.** Every cuda_raw lowering that takes a binary event operand calls
  `brainevent._op.util.cuda_event_abi(dtype, native_float=...)`, which returns
  `EventABI(suffix, prepare)`:

  | event dtype | suffix | prepare |
  |---|---|---|
  | bool | `_bool` | identity |
  | `native_float` (float32 by default; the weight dtype for `binary_fcnmm`) | `_float` | identity |
  | any other integer / floating dtype | `_bool` | `event_active` (`x != 0`) |
  | anything else | `TypeError` | |

  `prepare` runs only inside the lowering closure, so spike JVPs (straight-through, via the
  float operation with the raw tangent) are unaffected.
- **Errors.** Host-side failures in `.cu` files throw via `BE_CHECK` / `BE_CUDA_CHECK`; the
  kernix-generated wrapper turns them into an `xla::ffi::Error`, i.e. a Python exception.
  `abort()` is not used in host code. Device code keeps `__trap()`.

## Helpers (`brainevent/_op/util.py`)

```python
def event_active(x) -> jax.Array          # bool: x; else x != 0
def event_factor(x, *, dtype) -> jax.Array  # event_active(x).astype(dtype)
class EventABI(NamedTuple): suffix: str; prepare: Callable[[jax.Array], jax.Array]
def cuda_event_abi(dtype, *, native_float=jnp.float32) -> EventABI
```

`spike_suffix` (unused) is removed; `_csr/binary.py::_cuda_event_abi` is replaced.

## Tests

- `_op/util_test.py`: ABI table, `event_active`/`event_factor` on bool/int/float/NaN.
- Per format (CSR, CSR indexed, dense, FCN, JITC ×3, TCSR, plasticity, compact/bitpack):
  events `[2, -1, 0, 0.5]` (and NaN where meaningful) in bool/int8/uint8/int16/int32/f16/bf16/
  f32/f64, compared with `(x != 0) @ W`, on every available backend.
- GPU: the #199 (CSR f64, default backend) and #204 (dense f64) repros; a bad-dtype CUDA call
  raises a Python exception in a subprocess instead of aborting.
- x64: `csr_to_csc_index` emits no `FutureWarning` (#200, fixed in e18f81c).

## Migration note

Negative and NaN events used to be inactive on most backends and are now active. Users who
relied on `x > 0` thresholding should pass `x > 0` explicitly.

## Binary weight gradients (#198)

The forward pass of every binary operation multiplies weights by `event_factor(x)`. Its
derivatives must be the derivatives of that function:

- weight JVP: the binary primitive applied to the weight tangent (already the case);
- weight cotangent: built from `event_factor(x)`, never from raw `x`. Affected branches: CSR
  mv/mm hetero (+ indexed), dense mv/mm, FCN mm, JIT scalar mv/mm, JIT normal mv/mm (loc and
  scale). FCN mv, JIT uniform and TCSR `bptt` already comply; TCSR `pp_prop` is value-weighted
  by design and unchanged;
- event tangent / cotangent: unchanged straight-through estimator `W^T ct`.

`brainevent._test_util.check_event_op_grads(op, dense_op, params, x, mode=...)` checks forward,
reverse mode (weights and events), the JVP/VJP dot-product identity and `vmap` against a dense
reference, with `x` containing `2.0`, `-1.0` and `0`. Every format's tests call it.
