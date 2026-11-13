# GPU cuSPARSE hetero correctness comparisons

## Goal

Add 2512 GPU-only pytest cases for a fixed 1024 by 1024 graph with
`round(0.1 * 1024**2)` unique edges, seed 0, and per-edge weights. MV uses one
sample; MM uses four distinct samples. CPU runs collect and skip every case.

## Decisions

- Compare CUDA and cuSPARSE within each case, element by element, and check an
  independent FP64 reference rounded to the result dtype. Use runtime arguments
  and inspect compiled calls so a fallback cannot masquerade as a native route.
- Cross four floating weight/cotangent dtypes with all 13 standard event dtypes,
  binary/weighted modes, and int32/int64 offsets. Canonical indices are int32;
  scope x64 to each test. Weighted bool is a valid public input.
- CSR direct MV/MM covers two directions (832); indexed MV/MM covers the real
  mirror scatter direction (416). TCSR public MV/MM covers two directions
  (832), including inherited CSR paths, with 16 additional native weighted MM
  direct/mirror cases. SDDMV and SDDMM cover the native transpose direction
  (208 each). Each case compares both backends; batch size is not a case count.
- Keep finite bounded inputs for genuine cuSPARSE execution. Existing special
  value/empty/long-row suites continue to cover those separate contracts.
- Fix tolerances by dtype and reduction length before running. Record maximum
  absolute/relative errors, failed element count, and worst position as JUnit
  properties. Diagnose failures against FP64; never automatically relax limits.
- JAX's CSR cuSPARSE lowering only handles FP32/FP64. The independent CSR
  oracle therefore quantizes half operands to their original storage type,
  accumulates through actual FP32 cuSPARSE, and rounds back to that type.
  Inspect custom calls to distinguish this from JAX's half-precision fallback.
  Sampled products already perform this promotion in their cuSPARSE FFI.

### Fixed acceptance thresholds

Each element must satisfy `abs(cuda - reference) <= atol + rtol * abs(reference)`.

| dtype | CSR/TCSR rtol and atol | sampled batch 4 rtol / atol |
| --- | --- | --- |
| FP16 | 3e-3 | 1e-3 / 1e-6 |
| BF16 | 3e-2 | 8e-3 / 1e-6 |
| FP32 | 1e-4 | 2e-6 / 1e-6 |
| FP64 | 1e-10 | 2e-14 / 1e-14 |

Sampled batch 1 is compared exactly. Thresholds are assertions, rather than a
claim that every production path currently meets them. In particular, CSR half
scatter kernels update half output atomically per edge; they can accumulate
more rounding error than the FP32 cuSPARSE oracle. Keep these failures visible.

## Implementation

1. Add shared deterministic inputs, GPU gating, scoped x64, references and error
   reporting in `brainevent/_csr/test/_test_util.py`.
2. Add `cusparse_comparison_test.py` in the CSR, TCSR and SDDMM test directories.
3. Verify exactly 2512 collected cases and CPU skips, then run all GPU cases.

## Verification

Run the three new modules with `python -m pytest ... -m '' --junitxml=...` using
GPU 0 and disabled JAX preallocation:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest \
  brainevent/_csr/test/cusparse_comparison_test.py \
  brainevent/_tcsr/test/cusparse_comparison_test.py \
  brainevent/_sddmm/test/cusparse_comparison_test.py \
  -m '' -o junit_family=xunit1 --junitxml=cusparse-1k.xml
```

Repeat CPU execution with
`JAX_PLATFORMS=cpu`; every case must skip without allocating comparison inputs.
Run relevant existing integration tests and lint for changed Python files.
