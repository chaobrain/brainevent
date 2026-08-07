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
