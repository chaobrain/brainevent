# TCSR SDDMV（batch=1）与 cuSPARSE 对比基准

> **2026-10-16**：输入生成、计时循环、驱动与输出已按 ref 重组，见 [sddmv-benchmark-ref-layout.md](sddmv-benchmark-ref-layout.md)；本文第 3 节（方案与每一步的调用）仍然有效，其余部分为历史记录。

日期：2026-10-15。分支：`mm-optimizing`。由 [sddmm-cusparse-benchmark.md](sddmm-cusparse-benchmark.md) 9.1 节拆出。
计时口径于同日改为 CUDA C++ 计时，方案见 `brainevent/_tcsr/dev/sddmv/benchmark/timing_refactor_plan.md`。

## 1 目标

在独立目录 `brainevent/_tcsr/dev/sddmv` 中，用同一份输入比较四个 SDDMV 方案与 batch=1 的 `cusparseSDDMM` 每一步的 GPU 耗时。计时在 CUDA C++ 中完成，不经过 JAX 调用，benchmark 不校验输出。正确性由各方案自己的 pytest 负责。不依赖 `dev/sddmm` 的任何文件。

## 2 算子语义

方阵 `W` 为 `N × N` CSR，第 `s` 个槽位于 `(i, j)`。`a`、`b` 形状均为 `[N]`，输出 `C[nnz]` 为 float32，槽序与 CSR 一致：

- `binary_*`：`C[s] = [a[i] > 0] · b[j]`；
- `float`：`C[s] = a[i] · b[j]`；
- 非活动行为 0。

## 3 方案与每一步的调用

各方案的 `.cu` 原样 `#include` 进计时程序，每一步调用算子原本的入口函数，参数用 `BE::Tensor(ptr, shape, ndim, dtype)` 构造。

| scheme | 来源 | 每一步调用 | 窗口内包含 |
| --- | --- | --- | --- |
| `baseline` | `baseline/sddmv_{binary,float}.cu`，与 `brainevent/_tcsr/sddmm` 逐字节一致 | `tcsr_sddmv_dweight_{binary_f32_bool,binary_f32_float,float_f32}_t` | memset count、memset dW（nnz 长）、compact、panel × macro tile 数 |
| `sddmv_prezeroed` | `dev/sddmv-prezeroed/`（基线删掉 dW memset） | 同名入口，`dweight_in` 与 `dweight` 指向同一块 | memset count、compact、panel × macro tile 数 |
| `hybrid` | `dev/hybrid-sddmv/hybrid_sddmv.cu`（hybrid v1） | `hybrid_sddmv_binary_f32_{bool,float}`，`dweight_in` 与 `dweight` 同一块，scratch 与 output 同一块 | memset dW、memset status、extract、block |
| `hybrid_prezeroed` | `dev/hybrid-sddmv-prezeroed/` | 同 hybrid | memset status、extract、block |
| `cusparse` | `benchmark/cusparse_timing.cu` | `cusparseDnMatSetValues(A=a_k)`、`cusparseDnMatSetValues(B=ct_k)`、`cusparseSDDMM`，`alpha = 1`、`beta = 0`、ALG_DEFAULT，不调用 preprocess | SDDMM |

- 实现名为 `<scheme>_<binary_bool|binary_f32|float>`，hybrid 两个方案只有 binary，cuSPARSE 使用 float 那份输入，共 11 个实现。
- hybrid 编译时使用 `-DBE_HYBRID_TPR_THRESHOLD=128 -DBE_HYBRID_TASK_NNZ=4096`。`task_capacity = Σ_{nnz>128} ceil(nnz/4096)` 由 `run.py` 计算后传入。

## 4 输入与计时

- **结构**：随机流与 SDDMM 基准相同。结构数组留在 GPU 上，以只读设备指针交给 C++。
- **32 组用例**：第 `k` 步（`k = 0..31`）用 `build_dense(N, rate, (SEED, k))` 生成 `(a_k, b_k)`，各步互相独立，在窗口外堆叠成 `[32, N]` 放到 GPU 上。`b_k = standard_normal(N)`；事件在 `N` 个位置中无放回抽取 `round(rate·N)` 个。
- **缓冲**：dW 和 scratch 由 C++ 分配，分配时清零一次，之后从不重置，各步直接覆盖。
- **一轮**：在 non-blocking stream 上 `cudaEventRecord(start)` → 依次执行 32 步 → `cudaEventRecord(stop)` → 同步，每步耗时 = 窗口耗时 / 32。
- **轮数**：默认预热 3 轮、计时 20 轮。记录每轮的每步耗时，以及它们的 mean / std / min / max / median。
- **窗口外**：用例生成、缓冲分配、cuSPARSE 的 handle、描述符、bufferSize 和 buffer。
- **模式**：只用 stream 模式，逐个 launch，不录 CUDA Graph。

## 5 默认网格

`N ∈ {1e4, 1e5, 1e6}` × `density ∈ {1e-4, 1e-3, 1e-2, 1e-1}` × `event_rate ∈ {1e-4, 1e-3, 0.01, 0.1, 1.0}` × 11 个实现，共 660 条。N = 1e6、density ≥ 1e-2（nnz ≥ 1e10）超出内存预算和 int32 索引范围，记为 `skipped_memory`。

## 6 文件

```
brainevent/_tcsr/dev/sddmv/
    coverage.ini, mypy.ini
    baseline/sddmv.py, sddmv_binary.cu, sddmv_float.cu, test/
    benchmark/README.md, timing_refactor_plan.md, run.py, timing_common.cuh,
              tcsr_timing.cu, hybrid_timing.cu, cusparse_timing.cu, results/, test/
    dev/hybrid-sddmv/            hybrid_sddmv.{cu,py}, md/, results/, test/
    dev/hybrid-sddmv-prezeroed/  hybrid_sddmv_prezeroed.{cu,py}, test/
    dev/sddmv-prezeroed/         sddmv_{binary,float}.cu, sddmv_prezeroed.py, test/
    dev/block_sweep/, dev/md/    实验与方案文档
    ncu/
```

各方案的 `.py` 只供 pytest 使用（正确性测试，以及 benchmark 测试中与 C++ 输出逐位比较）。
