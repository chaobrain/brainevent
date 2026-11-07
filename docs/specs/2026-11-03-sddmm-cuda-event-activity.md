# CUDA SDDMM event activity

Use the native CUDA activity decision in both MV and MM. Weighted events are
cast to the cotangent dtype before the FFI call; values that become zero are
inactive, including when the corresponding cotangent is NaN or Inf. Remove
the Python/JAX finite scan and selection fallback from the CUDA adapters.
JAX backends retain activity checks on the original events. Binary dispatch
and the native dtype entry points remain unchanged.

Delete the unused selection helpers and call CUDA directly. Document the
backend-specific weighted activity rule and make the original-activity tests
explicitly select JAX. Add CUDA tests for mixed-dtype underflow, both leaves,
explicit/automatic dispatch, nonfinite cotangents and absence of a finite
scan in the lowering. Run the SDDMM tests on CPU/GPU and mypy.
