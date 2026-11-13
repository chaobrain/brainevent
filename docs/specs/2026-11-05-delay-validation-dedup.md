# Delay validation deduplication

## Goal

Remove repeated input checks from the JAX/CUDA delivery path while preserving
the standalone C++ entry and the delay object's head-assignment contract.

## Decisions

- Python primitive abstract evaluation owns shape/dtype checks; its shared JAX
  gate owns dynamic input checks. Remove the public call's eager head readback.
  Eager and JIT delivery now report dynamic errors during execution.
- Head assignment remains independently checked because clearing and reading
  a slot need not perform a delivery. Reuse one range predicate and check host
  wide integers before converting to a JAX/int32 representation.
- Reuse one static-flag validator at necessary Python boundaries. The public
  boundary remains necessary because JAX cache keys equate booleans with 0/1.
- CUDA FFI calls a prepared execution core directly. Standalone `launch_*`
  functions validate their own inputs once through a shared native preflight,
  then invoke that same core. Remove head/count/task/capacity checks from the
  delivery kernels; do not introduce a validation-mode flag or a second kernel
  implementation. Keep CUDA runtime and actual pointer-alias checks.
- The native API has no allocation lengths/nnz operand. Preserve its documented
  caller responsibility for CSR/column/offset/allocation bounds; native
  preflight is not advertised as the complete Python input validator.

## Steps and verification

1. Add wide-integer head-assignment regressions and preserve standalone native
   error tests; update the eager-delivery exception contract.
2. Deduplicate Python checks and split native preflight from execution. Include
   the new private header in the CUDA source cache hash.
3. Run CPU/GPU delay tests, standalone C++ linkage/error tests, JAX 0.9 checks,
   raw-bind sharding probes and mypy. Record results in the root review document.
