# Sparse apply, sum, and TCSR public contract

Status: implemented

## Scope

- Require `apply(fn)` to preserve the physical value-buffer shape for CSR,
  CSC, FixedNumPerPre, FixedNumPerPost, and TCSR. Dtype and unit changes remain
  supported.
- Make `sum(axis=None)` return the logical matrix sum for heterogeneous and
  homogeneous storage in the CSR/CSC, FCN, and TCSR families.
- Document TCSR sorted-source and x64 requirements across its public API.
- Complete NumPy-style documentation for the public TCSR data interface.

Row slicing, axis reductions, sparse solve, STDP implementation, dt2t behavior,
and matrix-multiplication kernels are outside this change.

## Apply invariant

`apply(fn)` evaluates `fn` once on the current physical data buffer. The result
must have exactly the same shape as that buffer. A shape mismatch raises
`ValueError`; dtype and physical unit may change. This prevents an apply call
from silently changing heterogeneous storage into homogeneous storage, or the
reverse.

## Sum invariant

Only `axis=None` is supported. Let `nse` be the number of represented sparse
entries:

- heterogeneous storage returns `data.sum()`;
- homogeneous size-one storage returns `data.sum() * nse`;
- any explicit axis continues to raise `NotImplementedError`.

The result must agree with `todense().sum()` for CSR, CSC, both FCN layouts,
TCSR, and transposed TCSR views. Units must be preserved.

## TCSR construction contract

TCSR requires JAX x64 support because canonical and mirror row pointers use
`int64`. Callers must enable `jax_enable_x64` before construction.

`TCSR(source)` and `TCSR.from_sorted_csr(source)` are trusted entry points. The
input must be a plain CSR whose column indices are nondecreasing within every
row. They preserve source entry order and do not verify sorting. Callers that
cannot guarantee this invariant must use `TCSR.fromcsr(source)`, which stably
sorts the indices and corresponding heterogeneous values.

## Verification

Regression tests run on GPU 1 and cover valid apply transforms, rejected shape
changes, homogeneous and heterogeneous sums, units, both FCN layouts, CSR/CSC,
and both logical TCSR orientations. Relevant suites and mypy must pass.
