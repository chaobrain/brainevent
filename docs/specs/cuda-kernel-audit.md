# CUDA kernel audit — correctness, dead code, performance

Status: in progress
Scope: full sweep of the 46 `.cu` files and 7 headers shipped as package data, looking for
(a) memory-safety and correctness defects, (b) dead code the previous cleanup missed, and
(c) performance headroom.

Predecessor: [`cuda-operator-cleanup.md`](cuda-operator-cleanup.md), which removed 102
unreachable `@BE` entry points, 2 orphan headers and 8 unused `cuda_common.h` symbols. That
pass audited *reachability*; it did not look inside the kernels.

## Validation environment

RTX 3080 Ti Laptop GPU (sm_86), driver 596.49, CUDA 13.2 / nvcc 13.1, jax 0.11.0 with
jaxlib CUDA. `compute-sanitizer` is **unavailable** on this host (WSL2/WDDM: "Failed to
initialize WDDM debugger interface / Device not supported"), so memory-safety findings are
established with purpose-built canary kernels and standalone harnesses instead.

### Validation pitfall (cost an hour, worth recording)

`brainevent` is installed in editable mode pointing at the **main checkout**. A script run
as `python /tmp/foo.py` puts `/tmp` — not the CWD — on `sys.path[0]`, so it imports
`/mnt/d/codes/projects/brainevent/brainevent`, *not* the worktree. Early "the fix does not
work" results were actually the unfixed main checkout being re-tested. Every GPU validation
command must pass:

```
PYTHONPATH=<worktree-root> python ...
```

and the harness should print `brainevent.__file__` to prove which tree it exercised.
Deleting `~/.cache/brainevent` between runs is also required, because compiled modules are
keyed on a content hash of the `.cu` source plus flags.

## Finding 1 — CSRMM block kernels under-request dynamic shared memory (memory fault)

**Severity: high.** float64 `binary_csrmm` aborts the CUDA context with
`CUDA_ERROR_ILLEGAL_ADDRESS`; float32/float16/bfloat16 silently perform an
out-of-bounds shared-memory write that only survives by allocation-granularity luck.

### The defect

`_csrmm_nt_block_{homo,hetero}_kern` (and the `_perm_hetero` variant in the indexed file)
stage one accumulator per `(strip, lane)` pair:

```c
smem[strip * 32 + lane] = acc0 + acc1;   // strip in 0..7, lane in 0..31 -> 256 slots
__syncthreads();
if (strip == 0 && c < n) {
    for (int s = 0; s < 8; s++) sum += smem[s * 32 + lane];
```

The kernels launch with 256 threads, so this needs `8 * 32 * sizeof(ACC_T)` bytes. Both
files instead requested `8 * sizeof(ACC_T)`:

| File | Requested | Required | Instantiations |
|---|---|---|---|
| `_csr/binary_csrmm.cu` | `8 * sizeof(T)` | `8 * 32 * sizeof(T)` | 16 |
| `_csr/binary_indexed_csrmm.cu` | `8 * sizeof(T)` | `8 * 32 * sizeof(T)` | 10 |

`_csr/float_csrmm.cu` — the same kernel shape — already had the correct
`8 * 32 * sizeof(...)`, which is what identifies this as a dropped `* 32` rather than a
different design. The CSRMV block kernels reduce into `smem_red[warpid]` (8 slots), so
their `8 * sizeof(T)` is correct and was left alone.

### Why the bug is dtype-dependent

A canary kernel that requests `shm` bytes but writes `nf` floats per block, run with 65536
blocks so the SM is saturated, shows what the driver actually tolerates on sm_86:

```
request=32 bytes, 256 thr/block, 65536 blocks
  wrote   128 floats (   512 bytes): no error     corrupted=0
  wrote   256 floats (  1024 bytes): no error     corrupted=0
  wrote   512 floats (  2048 bytes): an illegal memory access was encountered
```

The per-block shared-memory window is rounded up to a granule large enough that a 1024-byte
overrun lands harmlessly, but 2048 bytes faults. That is exactly the split observed:

* `ACC_T = float` (f32/f16/bf16) writes 256 × 4 = **1024 bytes** → survives on this driver,
  still formally out of bounds and not portable.
* `ACC_T = double` (f64) writes 256 × 8 = **2048 bytes** → **faults**.

### Reproduction through the public API

`binary_csrmm(..., transpose=False, backend='cuda_raw')` with float64 weights and
`avg_nnz > 512` (the threshold at which `nt_auto` selects the block kernel). Same script,
same GPU, only the checkout differs:

```
main:                                       worktree (fixed):
  homo   float32 nt_warp : PASS               homo   float32 nt_warp : PASS
  homo   float32 nt_block: PASS               homo   float32 nt_block: PASS
  hetero float32 nt_warp : PASS               hetero float32 nt_warp : PASS
  hetero float32 nt_block: PASS               hetero float32 nt_block: PASS
  homo   float64 nt_warp : PASS               homo   float64 nt_warp : PASS
  homo   float64 nt_block: CRASH              homo   float64 nt_block: PASS
    CUDA_ERROR_ILLEGAL_ADDRESS                hetero float64 nt_warp : PASS
    (context destroyed, run aborts)           hetero float64 nt_block: PASS
```

Independently, a standalone harness that `#include`s the real `binary_csrmm.cu` and calls
`binary_csrmm_nt_auto_hetero_f64_bool` with hand-built `BE::Tensor` views reports
`no error` and `max abs err = 0` after the fix, and `an illegal memory access was
encountered` before it.

### Fix

Both files now pass `8 * 32 * sizeof(...)`, matching `float_csrmm.cu`.

### Regression test

`brainevent/_csr/binary_test.py::test_binary_csrmm_nt_block_shared_memory_is_large_enough`
(`@requires_gpu_backend`, `@pytest.mark.slow`, parametrised over homo/hetero) builds a CSR
matrix with 1024 nnz per row so `nt_auto` routes to the block kernel, and compares the
float64 `cuda_raw` result against `jax_raw`. Verified to **fail** (illegal memory access) with
the fix stashed and **pass** with it applied.

## Finding 2 — 52 unreachable `@BE` entry points remain

The predecessor pass reduced 749 → 647 annotations. A fresh reachability analysis — extract
every `// @BE <name>`, extract every `kernel_name` f-string template from non-test Python,
expand `{...}` to `[A-Za-z0-9_]*`, and match — leaves 52 entry points that no Python code
path can name.

Python only ever composes these CSR binary targets: `binary_csrm{v,m}_nt_auto{_homo,_hetero}`,
`binary_csrm{v,m}_nt_auto_perm_hetero`, `binary_csrm{v,m}_{wat,sraw}_hybrid*` and
`binary_indexed_csrm{v,m}_{wat,sraw}_hybrid_hetero*`. The transpose route moved to the
hybrid kernels, which stranded the whole `t_warp` family:

| File | Dead entry points | Count |
|---|---|---|
| `_csr/binary_csrmv.cu` | `binary_csrmv_t_warp_{homo,hetero}_*` | 16 |
| `_csr/binary_csrmm.cu` | `binary_csrmm_t_warp_{homo,hetero}_*` | 16 |
| `_csr/binary_indexed_csrmv.cu` | `binary_csrmv_t_warp_perm_hetero_*` | 8 |
| `_csr/binary_indexed_csrmm.cu` | `binary_csrmm_t_warp_perm_hetero_*` | 8 |
| `_csr/binary_indexed_csrmm.cu` | `binary_csrmm_nt_{warp,block}_perm_hetero_f32_*` | 4 |

Removing them also retires the `_t_warp_*_kern` device kernels, which nothing else launches.
Those kernels additionally call `atomicAdd` directly on `__half*` / `__nv_bfloat16*` rather
than through the arch-guarded `atomic_add_f16` / `atomic_add_bf16` helpers, so they would
fail to compile below sm_70 / sm_80 — retiring them removes that portability trap too.

`binary_indexed_test.py::test_indexed_mm_cuda_kernel_selects_perm_names` asserts the dead
names and their `DEFINE_` macros are present in the `.cu` text; those assertions pin dead
code and are updated with the removal.

## Finding 3 — CSRMM dense subscripts overflow 32-bit `int` (correctness)

The CSRMM kernels index the dense operands as `B[indices[j] * n + c]` and `C[row * n + c]`.
Both factors are runtime `int`, so the product is computed in 32-bit and wraps once
`k * n` (or `m * n`) exceeds 2^31. Demonstrated standalone at `k * n = 2148507648`:
`int` yields `-2146460672`, `size_t` yields `2148506624` — a wild pointer, not a slow path.

`float_csrmm.cu` had the same exposure. 30 subscripts across the four CSRMM sources were
widened by inserting a `(size_t)` cast on the leading factor; the smem strides
(`smem[strip * 32 + lane]`) were left alone since both factors there are block-bounded
constants. Full GPU CSR suite: 699 passed.

A tree-wide re-audit then found the same pattern in the JIT connectivity families:
`chunk_counts[row * n_chunks + chunk_id]` and `chunk_offsets[...]` in
`_jit_{normal,scalar,uniform}/{csr,dt2t}.cu` (24 sites). `n_chunks` defaults to 4, but
`chunk_size` is user-settable, so `row * n_chunks` is not bounded by anything the kernel
controls. Widened the same way. JIT GPU suites: 1512 passed, 4 skipped.

Reachability for both is the same and worth stating plainly: the index only overflows once
the indexed buffer itself exceeds 2^31 elements, i.e. ≥8.6 GB at 4 bytes (4.3 GB at fp16).
That is out of reach on a small card and reachable on an 80 GB one. The fix costs nothing
at runtime — the multiply widens to 64-bit, and none of these sites are in an inner loop —
so it is worth taking rather than documenting as a limit.

After this pass the only remaining unwidened multiplicative subscripts tree-wide are
shared-memory strides with a literal `* 32`, bounded by block size by construction.

## Finding 4 — CSRMV was missing its warp-per-row tier (performance)

### The defect

`float_csrmv.cu` documents and implements a three-tier row mapping — one thread per row
below `avg_nnz` 8, one warp per row up to 512, one block per row above. The three binary
CSRMV sources only had two tiers:

```c
if (avg_nnz <= 512) { thread-per-row } else { block-per-row }
```

so every row length from 16 to 512 — the normal range for sparse neural connectivity —
ran the thread-per-row kernel, where the 32 lanes of a warp each walk a *different* row
and every `indices[j]` load is uncoalesced.

Git history shows `DEFINE_CSRMV_NT_WARP_HOMO` existed at `da4a812~1`. The predecessor
cleanup deleted it as unreachable, which was true but was the wrong repair: the dispatcher
was what needed fixing.

### Measured, on sm_86, m = k = 65536, float32 hetero, bool spikes, 50 iterations

| avg_nnz | thread (ms) | warp (ms) | block (ms) | best |
|--------:|------------:|----------:|-----------:|:-----|
| 2 | 0.009 | 0.026 | 0.219 | thread |
| 8 | 0.017 | 0.034 | 0.278 | thread |
| 12 | 0.030 | 0.034 | 0.278 | thread |
| 16 | 0.040 | 0.034 | 0.276 | **warp** |
| 32 | 0.088 | 0.043 | 0.271 | **warp** |
| 64 | 0.209 | 0.076 | 0.272 | **warp** |
| 128 | 0.361 | 0.141 | 0.249 | **warp** |
| 256 | 0.950 | 0.277 | 0.325 | **warp** |
| 512 | 1.904 | 0.547 | 0.545 | block |

Up to **3.4x** on the tier that was missing. The crossovers give the thresholds used:
thread below 16, warp to 512, block above.

### Launch shape: pack warps, don't launch 32-thread blocks

The historical kernel (and `float_csrmv.cu` today) launched `<<<m, 32>>>` — one warp per
*block*. A block occupies a scheduler slot regardless of size, and an SM caps resident
blocks, so 32-thread blocks waste 7/8 of each slot. Packing 8 warps into a 256-thread
grid-strided block instead:

| avg_nnz | `<<<m, 32>>>` (ms) | 8 warps/block (ms) | speedup |
|--------:|-------------------:|-------------------:|--------:|
| 8 | 0.111 | 0.034 | 3.31x |
| 16 | 0.111 | 0.034 | 3.25x |
| 32 | 0.111 | 0.044 | 2.53x |
| 64 | 0.160 | 0.076 | 2.11x |
| 128 | 0.145 | 0.143 | 1.02x |
| 256 | 0.276 | 0.277 | 1.00x |
| 512 | 0.545 | 0.550 | 0.99x |

Strictly better at low row lengths, parity above. The same shape applied to the transpose
atomic-scatter kernel (`csrmv_t_warp`, launched unconditionally): 2.80x at `avg_nnz` 8,
1.81x at 16, parity through 256, and 0.94x at 512 — net strongly positive.

### Fix

- Added `DEFINE_CSRMV_NT_WARP_{HOMO,HETERO}` to `binary_csrmv.cu` and
  `DEFINE_CSRMV_NT_WARP_PERM_HETERO` to `binary_indexed_csrmv.cu`, instantiated for all
  weight/spike dtype pairs (24 new device kernels, **no new `@BE` entry points** — the
  `nt_auto` wrappers dispatch to them internally).
- Rewrote all three dispatchers as the three-tier ladder.
- Converted the two existing `float_csrmv.cu` warp kernels from `<<<m, 32>>>` to the packed
  grid-strided shape.
- Added `be_csrmv_warp_grid()` to `cuda_common.h` (shared by all three sources; clamps to
  `[1, 4096]` so an empty matrix cannot produce a zero-sized grid).

`row` is derived as `blockIdx.x * warps_per + warp_id`, making it **warp-uniform**. That is
what keeps the `__shfl_down_sync` reduction converged — the M17 precondition documented in
`cuda_common.h`. A lane-dependent row bound would silently corrupt the reduction.

### Verification

96 configurations through the public API (7 row lengths spanning all three tiers x 3 weight
dtypes x homo/hetero x bool/float spikes), `cuda_raw` against `jax_raw`:

- float32 and float64: agree to 1e-5 / 1e-12 relative on every tier.
- float16: differs from `jax_raw` by up to 3e-2 relative — *including on the block tier,
  which this change does not touch*. Resolved against an exact float64 ground truth: the
  CUDA path is **10-100x closer to exact** than `jax_raw` in all 8 cases (e.g. 2.4e-4 vs
  3.8e-2 at `avg_nnz` 1024). The kernels accumulate in float32; the reference accumulates
  in float16. This is a reference-precision artifact, not a kernel defect.

**End-to-end Python timing could not resolve the improvement on this machine**: JAX host
dispatch latency floors at ~1.2 ms for this call, well above the 0.03-0.5 ms of kernel time,
so the API-level A/B returns noise. The standalone CUDA benchmarks above are the measurement
of record.

### Regression tests

- `test_binary_csrmv_matches_reference_across_dispatch_tiers` — 7 row lengths straddling
  both thresholds x homo/hetero.
- `test_binary_csrmv_warp_tier_handles_ragged_and_empty_rows` — empty rows, a row count off
  the warps-per-block tiling, and wildly varying row lengths; also asserts empty rows are
  written rather than left uninitialised.
- `test_binary_csrmv_cuda_source_keeps_three_dispatch_tiers` — source-level, runs without a
  GPU. Necessary because a two-tier dispatcher is still *correct*, so no numeric test can
  catch the tier being dropped again. Verified to fail against the pre-change source.

## Finding 5 — `float_csrmm.cu` warp kernels kept the unpacked launch shape

`binary_csrmm.cu` and `binary_indexed_csrmm.cu` already launch their warp kernels as
`CSRMM_WARP_RPB` (4) rows per 128-thread block with a grid-strided loop and a capped
`grid.x`. `float_csrmm.cu` was never updated to match and still launched
`<<<dim3(m, c_blocks), 32>>>` — one warp per block — for both its NT and transpose warp
kernels. Same defect as Finding 4's launch shape, in the file that is otherwise the
reference implementation.

Measured on sm_86, m = k = 16384, n = 64, float32 homogeneous:

| avg_nnz | `<<<m, 32>>>` (ms) | packed 4/block (ms) | speedup |
|--------:|-------------------:|--------------------:|--------:|
| 2 | 0.062 | 0.030 | 2.05x |
| 4 | 0.055 | 0.031 | 1.77x |
| 8 | 0.059 | 0.038 | 1.58x |
| 16 | 0.088 | 0.062 | 1.42x |
| 32 | 0.142 | 0.111 | 1.28x |
| 64 | 0.251 | 0.211 | 1.19x |

A win across the entire range the warp tier serves (`avg_nnz <= 64`).

Both kernels were converted to the packed grid-strided form. The row guard becomes
`continue` rather than `return`, since returning inside a grid-strided loop would abandon
that warp's remaining rows — the same trap as in Finding 4.

`CSRMM_MAX_GRID_X` and `CSRMM_WARP_RPB` were duplicated verbatim in two `.cu` files;
rather than add a third copy they were moved to `cuda_common.h` alongside the CSRMV
geometry and the local definitions deleted.

## Audits that found nothing (recorded so they are not redone)

- **Dynamic shared-memory sizing, tree-wide.** 12 kernels use `extern __shared__`. Every
  request now matches the largest index the kernel can write. `_fcn/binary_fcnmm.cu` was
  the one to check closely — it uses the same `(warpid + stride) * 32 + lane` strip
  indexing that broke CSRMM — and it sizes correctly as
  `red_off + nwarps * 32 * ACC_SIZE`.
- **`binary_fcnmm.cu` tree reduction over a non-power-of-two warp count.** The reduction
  `for (stride = nwarps >> 1; stride > 0; stride >>= 1)` silently drops the top warps if
  `nwarps` is not a power of two. Not a live bug: all four launch sites hardcode
  `bsz = 256`, so `nwarps == 8`. Noted because a future block-size change would break it
  without any test failing.
- **Dispatch-tier gaps beyond Finding 4.** `dt2t.cu`, `plasticity_binary_on_{pre,post}.cu`
  and `slice_csr_slice_rows.cu` all already implement three-tier dispatch. The CSRMM
  `avg_nnz <= 512` two-tier ladders are *not* the Finding 4 defect: for SpMM the two tiers
  are warp and block (lanes map to columns), which is the right pair.
- **Warp-reduction convergence** and **`__syncthreads()` divergence**: all flagged kernels
  verified block- or warp-uniform.
- **Unguarded fp16/bf16 atomics** in `float_csrmm.cu`: compile fine on sm_75 under CUDA 13;
  sm_60 is not supported by the toolkit. Not a live bug.

## Edge cases considered

- **Orphaning a device kernel.** Removing an FFI wrapper can strand the `__global__` kernel
  it was the sole launcher of. `nt_warp_perm` / `nt_block_perm` device kernels stay: the
  `nt_auto_perm` dispatcher still launches them internally; only their standalone FFI
  wrappers are unreachable.
- **Shared-memory fix and occupancy.** Raising the request from 32 to 1024 bytes (2048 for
  f64) does not change occupancy on sm_86: at 256 threads/block the limit is
  1536/256 = 6 blocks/SM in both cases, confirmed with
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. So the fix costs nothing.
- **Cache invalidation.** Editing any `.cu` changes its content hash, so the first GPU run
  after this change recompiles. Expected.
- **CPU suite proves nothing.** `.cu` sources are never compiled unless a GPU kernel is
  actually lowered. Every change here must be GPU-validated, per the predecessor spec's
  takeaway.
- **Warp reduction convergence.** The new warp kernels derive `row` from
  `blockIdx.x * warps_per + warp_id` only. Had the grid-stride bound been lane-dependent,
  lanes would retire at different trip counts and `__shfl_down_sync` would read from exited
  lanes. Audited all 6 pre-existing warp-reduction kernels the same way; all are
  warp-uniform.
- **Empty rows in the warp kernel.** The thread and block kernels early-return on
  `start == end`; the warp kernel deliberately does not, because an early `return` inside a
  grid-strided loop would skip that warp's remaining rows. Instead the inner loop simply
  does not execute, the reduction yields `ACC_ZERO`, and lane 0 writes it. Covered by the
  ragged-row test.
- **Zero-sized grid.** `be_csrmv_warp_grid()` clamps to at least 1. `csrmv_t_warp` has no
  `avg_nnz` guard, so an `m == 0` matrix would otherwise reach the launch with a
  zero-sized grid — an invalid configuration. (The pre-change `<<<m, 32>>>` had the same
  hole; this closes it.)
- **Threshold asymmetry with `float_csrmv.cu`.** The binary sources switch to the warp tier
  at `avg_nnz` 16 while `float_csrmv.cu` uses 8. The measured crossover for the binary
  kernels is between 12 and 16 (0.030 vs 0.034 at 12; 0.040 vs 0.034 at 16). The float
  threshold was left at 8: converting its warp kernel to the packed shape only makes the
  warp tier faster, so an unchanged threshold stays on the safe side of the crossover.
