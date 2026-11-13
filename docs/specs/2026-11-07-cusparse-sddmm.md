# cuSPARSE SDDMM Backend

## Goal

Add an explicit GPU `backend='cusparse'` to the vector and matrix sampled-product
leaves without changing `auto`, `cuda_raw`, or differentiable `jax_raw` routing.

## Decisions

- Both leaves share `_common/cusparse_sddmm.cu`; the vector leaf is the same
  sampled dense product with contraction dimension one.
- Use the existing `@BE` FFI ABI, caller CUDA stream, checked CSR dtypes, and a
  JAX-zeroed output aliased to its input. Dense operands are row-major with the
  right operand stored transposed for cuSPARSE's preferred layout.
  At the FFI boundary, promote column indices to the row-offset width because
  CSR SDDMM requires matching index types; the public int32 column ABI stays.
- Run `cusparseSDDMM_bufferSize`, `cusparseSDDMM_preprocess`, and `cusparseSDDMM`
  for the current CSR arrays on each call. Do not cache sparsity-dependent plans.
  Any library workspace is private and allocated/freed on the caller stream.
  The initial implementation disables CUDA graph capture.
- Support CSR, both event directions, and float/binary/weighted modes. Preserve
  physical slot order, unsorted columns, and supported duplicate patterns.
  Reject COO and `nnz > rows * cols` explicitly rather than silently using JAX.
- Compute float16/bfloat16 inputs in float32 and restore the result dtype;
  preserve float64. Weighted events are cast to the cotangent dtype before
  testing activity, consistent with the native CUDA backend.
- Event calls with nonfinite cotangents use exact JAX selection so inactive
  events suppress NaN/Inf before reduction. Ordinary products use cuSPARSE
  directly. Empty slots and zero contraction dimensions return zeros.

## Steps

1. Reproduce missing GPU registrations in the existing CUDA test module.
2. Implement the shared native FFI and connect both leaf registrations/calls.
3. Add focused GPU and validation tests, then run neighboring regressions.

## Verification

Compare cuSPARSE with JAX for MV/MM, modes, directions, value/offset dtypes,
runtime sparsity changes, duplicates/empty rows, repeated calls, vmap, and
shard_map with finite/nonfinite cotangents. Check lowered IR contains the actual
cuSPARSE FFI target.
Verify unsupported formats and oversized duplicate patterns fail clearly and
existing automatic/hybrid paths still pass.