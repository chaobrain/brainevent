# TCSR SDDMM 与 cuSPARSE 对比基准

日期：2026-10-14。分支：`mm-optimizing`。

## 1 目标

比较 `brainevent/_tcsr/dev/sddmm` 中 binary 与 float 两个 SDDMM CUDA 算子和 cuSPARSE `cusparseSDDMM` 的完整 JAX 调用耗时，并校验输出。

## 2 算子语义

方阵 `W` 为 `N × N` CSR，第 `s` 个存储槽位于 `(i, j)`。稠密输入 `A` 形状为 `[N, k]`，`B` 形状为 `[k, N]`，均为行主序。所有实现输出 `C[nnz]` float32，槽序与 CSR 一致：

| 实现 | A dtype | 计算 |
| --- | --- | --- |
| `be_binary_bool` | bool | `C[s] = Σ_b [A[i,b]] · B[b,j]` |
| `be_binary_f32` | float32，活动值为 1.0 | `C[s] = Σ_b [A[i,b] > 0] · B[b,j]` |
| `be_float` | float32，带符号非零值 | `C[s] = Σ_b A[i,b] · B[b,j]`，内核跳过 `A[i,b] = 0` |
| `cusparse_binary` | float32，即 binary 的 0/1 矩阵 | `C = (A·B) ∘ spy(W)` |
| `cusparse_float` | float32，与 `be_float` 同一份 | 同上 |

`k` 即 SDDMM 内积维，取 BE 支持的 4、8、16、32、64、128、256、512。

## 3 入口形态

两侧共用同一块 `A`、`B` 显存；binary-bool 时 cuSPARSE 读取同一事件集合转出的 0/1 float32 矩阵。

**BE**：直接以 JAX FFI 调用 dev 源文件，不经发布版 Python 包装，因此计时内无转置。

- binary：`tcsr_sddmm_dweight_binary_f32_{bool,float}_t(A, B, local_targets, indptr, tile_offsets) -> (C, masks)`。
- float：`tcsr_sddmm_dweight_float_f32_t(A, B, local_targets, indptr, tile_offsets) -> (C, masks, first_phase, active_counts, active_rows)`。
- 结构来自 `TCSR.from_sorted_csr(csr)`：`indptr` 为 int64，`local_targets` 为 uint16，`tile_offsets` 为 int32。scratch 由 XLA 每次调用分配，计入耗时。

**cuSPARSE**：新增 `baseline/cusparse_sddmm.cu`，FFI 入口 `cusparse_sddmm_csr_f32(A, B, indptr, indices) -> C`。

- `matA`：`N × k`、`ld = k`、行主序，`opA = NON_TRANSPOSE`。
- `matB`：`k × N`、`ld = N`、行主序，`opB = NON_TRANSPOSE`。按文档这不是 CSR 快路径，但它与 BE 输入是同一份数据，本基准不额外转置。
- `matC`：CSR，`indptr`、`indices` 均为 int32，零基。
- `alpha = 1`，`beta = 0`，`computeType = CUDA_R_32F`，`CUSPARSE_SDDMM_ALG_DEFAULT`。
- 调用内先异步清零 `C`，避免未初始化值经 `beta·C` 传播 NaN；BE 内核同样自行清零。
- handle 进程内缓存；workspace 按 `cusparseSDDMM_bufferSize` 结果只增不减地缓存，扩容只发生在预热阶段。描述符每次调用创建与销毁（仅主机端开销）。
- 不调用 `cusparseSDDMM_preprocess`。

两侧均以 `allow_cuda_graph=False` 编译。

## 4 输入生成

每个 `(N, density, seed)` 生成一次 CSR，在其上遍历 `k` 与事件率：

- 行长均匀：`nnz = N²·density`，必须整除为每行 `dN` 个元素；每行无放回抽列并升序排序，int32 索引。
- `W` 的值不参与 SDDMM，固定为 1.0。
- `B`：每个 `k` 生成一次标准正态 float32。
- 事件：在 `N·k` 个位置中无放回均匀抽取 `round(rate·N·k)` 个活动位置，独立同分布；`A_bool` 为该集合，`A_f32 = A_bool` 转 0/1；float 用同一集合，值为 `±U(0.5, 1.5)`。
- 随机流：`SeedSequence(seed).spawn(4)` 依次用于列、B、事件位置、float 值。

## 5 正确性

GPU 上以 float64 计算参考值：`nnz ≤ 100_000` 时全量，否则抽样 100_000 个槽位，并加入首行和末行全部槽位。

误差界：`|C[s] − ref[s]| ≤ 2·k·ε₃₂·Σ_b |a_b·B[b,j]| + 1e-30`。另外要求 `be_binary_bool` 与 `be_binary_f32` 逐位相等。计时结束后再校验一次，确认重复调用仍正确。

## 6 计时

`jax_call_ms`：预热 `warmup` 次后，连续调用 `repeat` 次再同步，取平均值，与 `new_mv_operator` 基准口径相同。输入生成、TCSR 构建、编译与校验不计时。

## 7 默认网格（小网格）

| 维度 | 取值 |
| --- | --- |
| N | 10_000、100_000 |
| density | 1e-4、1e-3、1e-2、1e-1 |
| k | 4、8、16、32、64、128、256、512 |
| event_rate | 0.01、0.1、1.0 |
| seed | 0 |

共 8 种结构 × 8 种 k × 3 种事件率 × 5 种实现，即 960 条记录。超过显存或主机内存预算的组合记为 `skipped_memory`，保留在结果中。

## 8 输出

写入 `benchmark/results/<时间戳>/`：`manifest.json` 记录环境、源码摘要和状态计数；`results.jsonl` 每条记录一个实现，包含参数、`nnz`、实际活动数、正确性指标和计时。

## 9 文件

```
brainevent/_tcsr/dev/sddmm/
    coverage.ini, mypy.ini
    baseline/                 发布版 brainevent/_tcsr/sddmm 的 SDDMM dev 副本 + cusparse_sddmm.cu
    baseline/test/sddmm_test.py
    benchmark/README.md, benchmark/run.py
    benchmark/test/run_test.py
    dev/, ncu/
```

## 9.1 batch=1

已拆分为独立基准，见 [sddmv-cusparse-benchmark.md](sddmv-cusparse-benchmark.md)。本基准 `--batches` 只接受 4 到 512。

## 10 已知局限

- cuSPARSE 的 B 布局不在其 CSR 快路径上；如需上界，可后续增加预转置口径。
- 未使用 `cusparseSDDMM_preprocess`。
- BE 的列信息为 u16 `local_targets` 加 `tile_offsets`，cuSPARSE 为 i32 `indices`，属于格式差异。
- BE 的 `indptr` 为 int64（由 TCSR 构建），cuSPARSE 为 int32。
