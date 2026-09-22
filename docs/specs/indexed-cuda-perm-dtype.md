# Indexed CUDA Permutation Dtype

## Problem

The non-transposed raw CUDA indexed CSR kernels accepted an `int64` `indptr`
and Python aligned `perm` to that dtype, but the CUDA source read `perm` as an
`int32_t` pointer. This interpreted 64-bit permutation entries with the wrong
width and selected incorrect weight positions.

## Contract

For indexed heterogeneous CSR CUDA kernels, `perm.dtype` must equal
`indptr.dtype`. Both are signed int32 or int64 offset/index arrays. The CUDA
kernel must read `perm` through the dispatched `IndptrT` type.

## Scope

Apply the contract to the non-transposed raw CUDA MV and MM indexed kernels.
The existing Python normalization remains responsible for aligning `perm` to
`indptr`; the CUDA FFI entry points reject a mismatched ABI defensively.
